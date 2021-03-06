/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/hlo_module_group_metadata.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

string HloModuleGroupMetadata::TrackedInstruction::ToString() const {
  string repr =
      (instruction_ != nullptr) ? instruction_->ToShortString() : "NULL";
  switch (kind_) {
    case ComputationKind::kInvalid:
      repr += ":INVALID";
      break;
    case ComputationKind::kWhileCondition:
      repr += ":WHILE_CONDITION";
      break;
    case ComputationKind::kWhileBody:
      repr += ":WHILE_BODY";
      break;
    case ComputationKind::kConditionalTrue:
      repr += ":CONDITIONAL_TRUE";
      break;
    case ComputationKind::kConditionalFalse:
      repr += ":CONDITIONAL_FALSE";
      break;
  }
  return repr;
}

/* static */ StatusOr<std::unique_ptr<HloModuleGroupMetadata>>
HloModuleGroupMetadata::Build(const std::vector<HloModule*>& modules) {
  auto metadata = absl::make_unique<HloModuleGroupMetadata>(modules);
  TF_RETURN_IF_ERROR(metadata->Build());
  return std::move(metadata);
}

Status HloModuleGroupMetadata::Build() {
  TF_RETURN_IF_ERROR(RecordInstructions());
  TF_RETURN_IF_ERROR(VerifyChannelInstructions());

  // Record all companion while instructions.
  const auto visitor = [this](HloInstruction* hlo) -> Status {
    // We only need to process if the instruction is within the computation
    // of a companion instruction, like in the condition or body computation
    // of a While.
    const TrackedInstruction* tracked = GetTrackedInstruction(hlo->parent());
    if (tracked == nullptr) {
      return Status::OK();
    }
    // Add the parent computation of this channel instruction and its peer
    // computation (both must be while computations) as companions.
    if (IsChannelInstruction(hlo)) {
      HloComputation* peer_computation = PeerComputation(hlo);
      const TrackedInstruction* peer_tracked =
          GetTrackedInstruction(peer_computation);
      TF_RET_CHECK(peer_tracked != nullptr)
          << "Peer instruction is not a possible companion";
      TF_RET_CHECK(*tracked == *peer_tracked)
          << "Peer instruction does not match the computation kind";
      TF_RETURN_IF_ERROR(
          AddCompanion(tracked->instruction(), peer_tracked->instruction()));
    }

    // Add the parents of companion instructions (they must be all of the same
    // kind of instructions, opcode wise) as companions.
    if (IsCompanionInstruction(hlo)) {
      for (HloInstruction* companion : Companions(hlo)) {
        const TrackedInstruction* companion_tracked =
            GetTrackedInstruction(companion->parent());
        TF_RET_CHECK(companion_tracked != nullptr);
        TF_RET_CHECK(*tracked == *companion_tracked);
        TF_RETURN_IF_ERROR(AddCompanion(tracked->instruction(),
                                        companion_tracked->instruction()));
      }
    }
    return Status::OK();
  };

  // Visit the computations in postorder so that the companion information grows
  // from inner computations to outer ones.
  for (HloModule* module : modules_) {
    for (HloComputation* computation : module->MakeComputationPostOrder()) {
      TF_RETURN_IF_ERROR(computation->Accept(visitor));
    }
  }
  return Status::OK();
}

bool HloModuleGroupMetadata::IsChannelInstruction(
    const HloInstruction* instruction) const {
  switch (instruction->opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone:
      return true;
    default:
      return false;
  }
}

bool HloModuleGroupMetadata::IsCompanionInstruction(HloInstruction* hlo) const {
  return companion_set_index_.count(hlo) > 0;
}

bool HloModuleGroupMetadata::InstructionCommunicates(
    HloInstruction* hlo) const {
  return IsChannelInstruction(hlo) || IsCompanionInstruction(hlo);
}

const HloModuleGroupMetadata::Channel& HloModuleGroupMetadata::GetChannel(
    int64 channel_id) const {
  CHECK(channel_id_map_.find(channel_id) != channel_id_map_.end());
  return channels_[channel_id_map_.at(channel_id)];
}

HloComputation* HloModuleGroupMetadata::PeerComputation(
    const HloInstruction* instruction) const {
  CHECK(IsChannelInstruction(instruction));
  const Channel& channel = GetChannel(instruction->channel_id());
  switch (instruction->opcode()) {
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
      return channel.recv->parent();
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
      return channel.send->parent();
    default:
      LOG(FATAL) << "opcode not supported";
  }
}

std::vector<HloModuleGroupMetadata::TrackedInstruction>
HloModuleGroupMetadata::GetCompanionsPath(const HloInstruction* hlo) const {
  std::vector<TrackedInstruction> path;
  const HloComputation* parent = hlo->parent();
  const TrackedInstruction* companion;
  while ((companion = GetTrackedInstruction(parent)) != nullptr) {
    parent = companion->instruction()->parent();
    path.push_back(*companion);
  }
  return path;
}

bool HloModuleGroupMetadata::CheckCompanionPathsCompatibility(
    const std::vector<TrackedInstruction>& path0,
    const std::vector<TrackedInstruction>& path1) const {
  if (path0.size() != path1.size()) {
    VLOG(5) << "Companion path size do not match: " << path0.size()
            << " != " << path1.size();
    return false;
  }
  for (int64 i = 0; i < path0.size(); ++i) {
    if (path0[i] != path1[i]) {
      VLOG(5) << "Companion instructions at path index " << i
              << " do not have the same opcode: " << path0[i].ToString()
              << " vs " << path1[i].ToString();
      return false;
    }
  }
  return true;
}

int64 HloModuleGroupMetadata::GetModuleId(const HloModule* module) const {
  for (int64 i = 0; i < modules_.size(); ++i) {
    if (modules_[i] == module) {
      return i;
    }
  }
  LOG(FATAL) << "unknown module";
}

Status HloModuleGroupMetadata::RecordInstructions() {
  const auto visitor = [this](HloInstruction* hlo) -> Status {
    if (hlo->opcode() == HloOpcode::kWhile) {
      tracked_instructions_[hlo->while_condition()] =
          TrackedInstruction(hlo, ComputationKind::kWhileCondition);
      tracked_instructions_[hlo->while_body()] =
          TrackedInstruction(hlo, ComputationKind::kWhileBody);
    } else if (hlo->opcode() == HloOpcode::kConditional) {
      tracked_instructions_[hlo->true_computation()] =
          TrackedInstruction(hlo, ComputationKind::kConditionalTrue);
      tracked_instructions_[hlo->false_computation()] =
          TrackedInstruction(hlo, ComputationKind::kConditionalFalse);
    }
    if (!IsChannelInstruction(hlo)) {
      return Status::OK();
    }

    // Add a new channel if needed.
    if (channel_id_map_.find(hlo->channel_id()) == channel_id_map_.end()) {
      channels_.emplace_back();
      channels_.back().id = hlo->channel_id();
      channel_id_map_[hlo->channel_id()] = channels_.size() - 1;
      max_channel_id_ = std::max(max_channel_id_, hlo->channel_id());
    }
    Channel& channel = channels_[channel_id_map_[hlo->channel_id()]];

    if (hlo->opcode() == HloOpcode::kSend) {
      TF_RET_CHECK(channel.send == nullptr)
          << "channel id " << hlo->channel_id()
          << " is used by multiple send instructions";
      channel.send = hlo;
    }
    if (hlo->opcode() == HloOpcode::kRecv) {
      TF_RET_CHECK(channel.recv == nullptr)
          << "channel id " << hlo->channel_id()
          << " is used by multiple recv instructions";
      channel.recv = hlo;
    }
    if (hlo->opcode() == HloOpcode::kSendDone) {
      TF_RET_CHECK(channel.send_done == nullptr)
          << "channel id " << hlo->channel_id()
          << " is used by multiple send-done instructions";
      channel.send_done = hlo;
    }
    if (hlo->opcode() == HloOpcode::kRecvDone) {
      TF_RET_CHECK(channel.recv_done == nullptr)
          << "channel id " << hlo->channel_id()
          << " is used by multiple recv-done instructions";
      channel.recv_done = hlo;
    }
    return Status::OK();
  };

  for (HloModule* module : modules_) {
    for (auto* computation : module->computations()) {
      TF_RETURN_IF_ERROR(computation->Accept(visitor));
    }
  }
  return Status::OK();
}

Status HloModuleGroupMetadata::AddCompanion(HloInstruction* instruction1,
                                            HloInstruction* instruction2) {
  TF_RET_CHECK(instruction1->opcode() == HloOpcode::kWhile ||
               instruction1->opcode() == HloOpcode::kConditional);
  VLOG(2) << "adding as companions:" << instruction1->ToString() << " and "
          << instruction2->ToString();

  if (!ContainsKey(companion_set_index_, instruction1) &&
      !ContainsKey(companion_set_index_, instruction2)) {
    companion_sets_.push_back(
        absl::make_unique<std::unordered_set<HloInstruction*>>());
    auto companion_set = companion_sets_.back().get();
    companion_set->insert(instruction1);
    companion_set->insert(instruction2);
    companion_set_index_[instruction1] = companion_sets_.size() - 1;
    companion_set_index_[instruction2] = companion_sets_.size() - 1;
  } else if (!ContainsKey(companion_set_index_, instruction1)) {
    companion_sets_[companion_set_index_[instruction2]]->insert(instruction1);
    companion_set_index_[instruction1] = companion_set_index_[instruction2];
  } else if (!ContainsKey(companion_set_index_, instruction2)) {
    companion_sets_[companion_set_index_[instruction1]]->insert(instruction2);
    companion_set_index_[instruction2] = companion_set_index_[instruction1];
  } else if (companion_set_index_[instruction1] !=
             companion_set_index_[instruction2]) {
    companion_sets_[companion_set_index_[instruction1]]->insert(
        Companions(instruction2).begin(), Companions(instruction2).end());
    int64 index_to_remove = companion_set_index_[instruction2];
    for (HloInstruction* hlo : Companions(instruction2)) {
      companion_set_index_[hlo] = companion_set_index_[instruction1];
    }
    companion_sets_.erase(companion_sets_.begin() + index_to_remove);
  }
  return Status::OK();
}

Status HloModuleGroupMetadata::VerifyChannelInstructions() {
  for (const Channel& channel : channels_) {
    if (channel.send == nullptr) {
      return FailedPrecondition("missing send for id : %lld", channel.id);
    }
    if (channel.recv == nullptr) {
      return FailedPrecondition("missing recv for id : %lld", channel.id);
    }
    if (channel.send_done == nullptr) {
      return FailedPrecondition("missing send-done for id : %lld", channel.id);
    }
    if (channel.recv_done == nullptr) {
      return FailedPrecondition("missing recv-done for id : %lld", channel.id);
    }
  }

  // Check if the shapes match for each channel.
  for (const Channel& channel : channels_) {
    const Shape& send_shape = channel.send->operand(0)->shape();
    const Shape& recv_shape = channel.recv_done->shape();
    if (!ShapeUtil::Compatible(send_shape, recv_shape)) {
      return FailedPrecondition("send/recv shapes do not match");
    }
    const HloModule* send_module = channel.send->parent()->parent();
    const HloModule* send_done_module = channel.send_done->parent()->parent();
    if (send_module != send_done_module) {
      return FailedPrecondition(
          "send and send-done (channel=%lld) must be on the same device: %lld "
          "vs. %lld",
          channel.id, GetModuleId(send_module), GetModuleId(send_done_module));
    }
    const HloModule* recv_module = channel.recv->parent()->parent();
    const HloModule* recv_done_module = channel.recv_done->parent()->parent();
    if (recv_module != recv_done_module) {
      return FailedPrecondition(
          "recv and recv-done (channel=%lld) must be on the same device: %lld "
          "vs. %lld",
          channel.id, GetModuleId(recv_module), GetModuleId(recv_done_module));
    }
    if (send_module == recv_module) {
      return FailedPrecondition(
          "send and recv (channel=%lld) must be on different devices: %lld",
          channel.id, GetModuleId(send_module));
    }
  }

  // Check if channel instructions are used only in allowed computations.
  const auto allowed = [this](HloInstruction* hlo) {
    HloComputation* computation = hlo->parent();
    const HloModule* module = computation->parent();
    if (module->entry_computation() == computation ||
        tracked_instructions_.count(computation) > 0) {
      return true;
    }
    return false;
  };
  for (const Channel& channel : channels_) {
    if (!allowed(channel.send) || !allowed(channel.send_done) ||
        !allowed(channel.recv) || !allowed(channel.recv_done)) {
      return FailedPrecondition("channel is used in disallowed computation");
    }
  }
  // Check if the nest levels match for each channel.
  for (const Channel& channel : channels_) {
    std::vector<TrackedInstruction> path = GetCompanionsPath(channel.send);
    if (!CheckCompanionPathsCompatibility(
            path, GetCompanionsPath(channel.send_done)) ||
        !CheckCompanionPathsCompatibility(path,
                                          GetCompanionsPath(channel.recv)) ||
        !CheckCompanionPathsCompatibility(
            path, GetCompanionsPath(channel.recv_done))) {
      return FailedPrecondition(
          "Nest companion paths do not match for channel %lld", channel.id);
    }
  }
  return Status::OK();
}

}  // namespace xla
