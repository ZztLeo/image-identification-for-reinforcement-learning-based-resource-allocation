/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/costs/virtual_scheduler.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
// Class for testing virtual scheduler.
class TestVirtualScheduler : public VirtualScheduler {
 public:
  TestVirtualScheduler(const GrapplerItem* grappler_item,
                       const bool use_static_shapes, Cluster* cluster)
      : VirtualScheduler(grappler_item, use_static_shapes, cluster,
                         &ready_node_manager_) {}

  FRIEND_TEST(VirtualSchedulerTest, CalculateOutputSize);
  FRIEND_TEST(VirtualSchedulerTest, MemoryUsage);
  FRIEND_TEST(VirtualSchedulerTest, ControlDependency);
  FRIEND_TEST(VirtualSchedulerTest, ComplexDependency);
  FRIEND_TEST(VirtualSchedulerTest, Variable);
  FRIEND_TEST(VirtualSchedulerTest, InterDeviceTransfer);

 protected:
  FirstReadyManager ready_node_manager_;
};

class VirtualSchedulerTest : public ::testing::Test {
 protected:
  NodeDef node1_, node2_, node3_, node4_, node5_, node6_;
  std::unordered_map<const NodeDef*, NodeState> node_states_;

  // Device names:
  const string kCPU0 = "/job:localhost/replica:0/task:0/cpu:0";
  const string kCPU1 = "/job:localhost/replica:0/task:0/cpu:1";
  const string kChannelFrom0To1 = "Channel from CPU0 to CPU1";
  const string kChannelFrom1To0 = "Channel from CPU1 to CPU0";
  // Op names:
  const string kSend = "_Send";
  const string kRecv = "_Recv";
  const string kConv2D = "Conv2D";

  DeviceProperties GetDummyCPUDevice() {
    // Create CPU with 2 cores, 4 Ghz freq, 2 GB/s mem bandwidth.
    // - 8 Gflops
    // - 2 GB/s
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_frequency(4000);
    cpu_device.set_num_cores(2);
    cpu_device.set_bandwidth(2000000);
    return cpu_device;
  }

  void NodeSetUp(const string& name, const string& op_name,
                 const string& device_name, const uint64 time_ready,
                 NodeDef* node) {
    node->set_name(name);
    node->set_op(op_name);
    node->set_device(device_name);

    node_states_[node] = NodeState();
    node_states_[node].time_ready = time_ready;
    node_states_[node].device_name = device_name;
  }

  void SetUp() override {
    // node1_ to node6_ on kCPU0, with time_ready in reverse_order.
    NodeSetUp("Node1", kConv2D, kCPU0, 6000, &node1_);
    NodeSetUp("Node2", kConv2D, kCPU0, 5000, &node2_);
    NodeSetUp("Node3", kConv2D, kCPU0, 4000, &node3_);
    NodeSetUp("Node4", kConv2D, kCPU0, 3000, &node4_);
    NodeSetUp("Node5", kConv2D, kCPU0, 2000, &node5_);
    NodeSetUp("Node6", kConv2D, kCPU0, 1000, &node6_);

    // Initializes cluster_ and placer_.
    std::unordered_map<string, DeviceProperties> devices;

    // Set some dummy CPU properties
    DeviceProperties cpu_device = GetDummyCPUDevice();

    // IMPORTANT: Device is not actually ever used in the test case since
    // force_cpu_type is defaulted to "Haswell"
    devices[kCPU0] = cpu_device;
    devices[kCPU1] = cpu_device;
    cluster_.reset(new VirtualCluster(devices));
    placer_.reset(new VirtualPlacer(cluster_.get()));
  }

  // Three Conv2Ds with only two in fetch nodes.
  void CreateGrapplerItemWithConv2Ds() {
    Scope s = Scope::NewRootScope().WithDevice(kCPU0);
    auto x = ops::RandomUniform(
        s.WithOpName("x"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto y = ops::RandomUniform(
        s.WithOpName("y"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto z = ops::RandomUniform(
        s.WithOpName("z"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto f = ops::RandomUniform(
        s.WithOpName("f"), {kernel_, kernel_, depth_in_, depth_out_}, DT_FLOAT);
    std::vector<int> strides = {1, 1, 1, 1};
    auto c0 = ops::Conv2D(s.WithOpName("c0"), x, f, strides, "SAME");
    auto c1 = ops::Conv2D(s.WithOpName("c1"), y, f, strides, "SAME");
    auto c2 = ops::Conv2D(s.WithOpName("c2"), z, f, strides, "SAME");
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_conv2d_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"c0", "c1"};

    dependency_["c0"] = {"x", "f"};
    dependency_["c1"] = {"y", "f"};
  }

  // A Conv2D with a variable.
  void CreateGrapplerItemWithConv2DAndVariable() {
    Scope s = Scope::NewRootScope().WithDevice(kCPU0);
    auto x = ops::RandomUniform(
        s.WithOpName("x"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto f = ops::Variable(s.WithOpName("f"),
                           {kernel_, kernel_, depth_in_, depth_out_}, DT_FLOAT);
    std::vector<int> strides = {1, 1, 1, 1};
    auto y = ops::Conv2D(s.WithOpName("y"), x, f, strides, "SAME");
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_conv2d_var_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"y"};

    dependency_["y"] = {"x", "f"};
  }

  void CreateGrapplerItemWithMatmulChain() {
    Scope s = Scope::NewRootScope().WithDevice(kCPU0);
    // Add control dependencies to ensure tests do not rely on specific
    // manager and the order remains consistent for the test.
    auto a = ops::RandomUniform(s.WithOpName("a"), {3200, 3200}, DT_FLOAT);
    auto b = ops::RandomUniform(s.WithOpName("b").WithControlDependencies(a),
                                {3200, 3200}, DT_FLOAT);
    auto c = ops::RandomUniform(s.WithOpName("c").WithControlDependencies(b),
                                {3200, 3200}, DT_FLOAT);
    auto d = ops::RandomUniform(s.WithOpName("d").WithControlDependencies(c),
                                {3200, 3200}, DT_FLOAT);
    auto e = ops::RandomUniform(s.WithOpName("e").WithControlDependencies(d),
                                {3200, 3200}, DT_FLOAT);

    auto ab = ops::MatMul(s.WithOpName("ab").WithControlDependencies(e), a, b);
    auto abc = ops::MatMul(s.WithOpName("abc"), ab, c);
    auto abcd = ops::MatMul(s.WithOpName("abcd"), abc, d);
    auto abcde = ops::MatMul(s.WithOpName("abcde"), abcd, e);

    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_matmul_sequence_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"abcde"};

    dependency_["ab"] = {"a", "b"};
    dependency_["abc"] = {"ab", "c"};
    dependency_["abcd"] = {"abc", "d"};
    dependency_["abcde"] = {"abcd", "e"};
  }

  // AddN that takes 4 tensors with 10x10x10x10.
  void CreateGrapplerItemWithAddN() {
    Scope s = Scope::NewRootScope().WithDevice(kCPU0);
    auto x = ops::RandomUniform(s.WithOpName("x"), {10, 10, 10, 10}, DT_FLOAT);
    auto y = ops::RandomUniform(s.WithOpName("y"), {10, 10, 10, 10}, DT_FLOAT);
    auto z = ops::RandomUniform(s.WithOpName("z"), {10, 10, 10, 10}, DT_FLOAT);
    auto w = ops::RandomUniform(s.WithOpName("w"), {10, 10, 10, 10}, DT_FLOAT);
    OutputList input_tensors = {x, y, z, w};
    auto out = ops::AddN(s.WithOpName("out"), input_tensors);
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_addn_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"out"};

    dependency_["out"] = {"x", "y", "z", "w"};
  }

  // Graph with some placeholder feed nodes that are not in the fetch fan-in.
  void CreateGrapplerItemWithUnnecessaryPlaceholderNodes() {
    Scope s = Scope::NewRootScope().WithDevice(kCPU0);
    auto unnecessary = ops::Placeholder(s.WithOpName("unnecessary"), DT_FLOAT);
    auto x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT);

    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_extra_placeholders";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"x"};

    // Grappler Item Builder puts all placeholder nodes into the feed
    // list by default.
    grappler_item_->feed = {{"x", Tensor()}, {"unnecessary", Tensor()}};
  }

  // NoOp that takes 7 NoOps as control dependency.
  void CreateGrapplerItemWithControlDependency() {
    Scope s = Scope::NewRootScope().WithDevice(kCPU0);
    std::vector<string> input_noop_names = {"x", "y", "z", "w", "u", "v", "t"};
    std::vector<Operation> input_tensors;
    for (const auto& input : input_noop_names) {
      auto x = ops::NoOp(s.WithOpName(input));
      input_tensors.push_back(x.operation);
    }
    auto out =
        ops::NoOp(s.WithControlDependencies(input_tensors).WithOpName("out"));
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_control_dependency_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"out"};

    dependency_["out"] = input_noop_names;
  }

  // FusedBN [an op with multiple outputs] with multiple consumers (including
  // control dependency).
  void CreateGrapplerItemWithBatchNorm() {
    Scope s = Scope::NewRootScope().WithDevice(kCPU0);
    auto x = ops::RandomUniform(
        s.WithOpName("x"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto scale =
        ops::RandomUniform(s.WithOpName("scale"), {depth_in_}, DT_FLOAT);
    auto offset =
        ops::RandomUniform(s.WithOpName("offset"), {depth_in_}, DT_FLOAT);
    auto mean = ops::RandomUniform(s.WithOpName("mean"), {0}, DT_FLOAT);
    auto var = ops::RandomUniform(s.WithOpName("var"), {0}, DT_FLOAT);

    auto batch_norm = ops::FusedBatchNorm(
        s.WithOpName("bn"), x, scale, offset, mean, var,
        ops::FusedBatchNorm::IsTraining(true).Epsilon(0.1f));
    auto y = batch_norm.y;
    auto batch_mean = batch_norm.batch_mean;
    auto batch_var = batch_norm.batch_variance;

    auto z1 = ops::Add(s.WithOpName("z1"), x, y);
    auto z2 = ops::Add(s.WithOpName("z2"), batch_var, batch_var);
    auto z3 = ops::Add(s.WithOpName("z3"), batch_var, batch_var);
    std::vector<Operation> input_tensors = {
        batch_mean.op(),
        z1.z.op(),
        z2.z.op(),
        z3.z.op(),
    };
    auto z4 = ops::NoOp(s.WithControlDependencies(batch_var).WithOpName("z4"));

    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_complex_dependency_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"z1", "z2", "z3", "z4"};

    dependency_["bn"] = {"x", "scale", "offset", "mean", "var"};
    dependency_["z1"] = {"x", "bn"};
    dependency_["z2"] = {"bn"};
    dependency_["z3"] = {"bn"};
    dependency_["z4"] = {"bn"};
  }

  void CreateGrapplerItemWithSendRecv() {
    const string gdef_ascii = R"EOF(
node {
  name: "Const"
  op: "Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 3.1415
      }
    }
  }
}
node {
  name: "Send"
  op: "_Send"
  input: "Const"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "client_terminated"
    value {
      b: false
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:localhost/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:localhost/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 0
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "test"
    }
  }
}
node {
  name: "Recv"
  op: "_Recv"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "client_terminated"
    value {
      b: false
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:localhost/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:localhost/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 0
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "test"
    }
  }
  attr {
    key: "tensor_type"
    value {
      type: DT_FLOAT
    }
  }
}
library {
}
versions {
  producer: 24
}
    )EOF";

    grappler_item_.reset(new GrapplerItem);
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii,
                                                &grappler_item_->graph));
    grappler_item_->id = "test_graph";
    grappler_item_->fetch = {"Recv"};
  }

  void CreateGrapplerItemWithRecvWithoutSend() {
    const string gdef_ascii = R"EOF(
node {
  name: "Recv"
  op: "_Recv"
  device: "/job:localhost/replica:0/task:0/device:CPU:0"
  attr {
    key: "client_terminated"
    value {
      b: false
    }
  }
  attr {
    key: "recv_device"
    value {
      s: "/job:localhost/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device"
    value {
      s: "/job:localhost/replica:0/task:0/device:CPU:0"
    }
  }
  attr {
    key: "send_device_incarnation"
    value {
      i: 0
    }
  }
  attr {
    key: "tensor_name"
    value {
      s: "test"
    }
  }
  attr {
    key: "tensor_type"
    value {
      type: DT_FLOAT
    }
  }
}
library {
}
versions {
  producer: 24
}
    )EOF";

    grappler_item_.reset(new GrapplerItem);
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii,
                                                &grappler_item_->graph));
    grappler_item_->id = "test_graph";
    grappler_item_->fetch = {"Recv"};
  }

  // A simple while loop
  void CreateGrapplerItemWithLoop() {
    // Test graph produced in python using:
    /*
      with tf.Graph().as_default():
      i0 = tf.constant(0)
      m0 = tf.ones([2, 2])
      c = lambda i, m: i < 10
      b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
      r = tf.while_loop(
      c, b, loop_vars=[i0, m0],
      shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])
      with open('/tmp/graph.pbtxt', 'w') as f:
      f.write(str(tf.get_default_graph().as_graph_def()))
    */
    const string gdef_ascii = R"EOF(
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "ones"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "while/Enter"
  op: "Enter"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "frame_name"
    value {
      s: "while/while/"
    }
  }
  attr {
    key: "is_constant"
    value {
      b: false
    }
  }
  attr {
    key: "parallel_iterations"
    value {
      i: 10
    }
  }
}
node {
  name: "while/Enter_1"
  op: "Enter"
  input: "ones"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "frame_name"
    value {
      s: "while/while/"
    }
  }
  attr {
    key: "is_constant"
    value {
      b: false
    }
  }
  attr {
    key: "parallel_iterations"
    value {
      i: 10
    }
  }
}
node {
  name: "while/Merge"
  op: "Merge"
  input: "while/Enter"
  input: "while/NextIteration"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Merge_1"
  op: "Merge"
  input: "while/Enter_1"
  input: "while/NextIteration_1"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "while/Less/y"
  op: "Const"
  input: "^while/Merge"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "while/Less"
  op: "Less"
  input: "while/Merge"
  input: "while/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/LoopCond"
  op: "LoopCond"
  input: "while/Less"
}
node {
  name: "while/Switch"
  op: "Switch"
  input: "while/Merge"
  input: "while/LoopCond"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@while/Merge"
      }
    }
  }
}
node {
  name: "while/Switch_1"
  op: "Switch"
  input: "while/Merge_1"
  input: "while/LoopCond"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@while/Merge_1"
      }
    }
  }
}
node {
  name: "while/Identity"
  op: "Identity"
  input: "while/Switch:1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Identity_1"
  op: "Identity"
  input: "while/Switch_1:1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "while/add/y"
  op: "Const"
  input: "^while/Identity"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "while/add"
  op: "Add"
  input: "while/Identity"
  input: "while/add/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/concat/axis"
  op: "Const"
  input: "^while/Identity"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "while/concat"
  op: "ConcatV2"
  input: "while/Identity_1"
  input: "while/Identity_1"
  input: "while/concat/axis"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration"
  op: "NextIteration"
  input: "while/add"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration_1"
  op: "NextIteration"
  input: "while/concat"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "while/Exit"
  op: "Exit"
  input: "while/Switch"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Exit_1"
  op: "Exit"
  input: "while/Switch_1"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 21
}
  )EOF";

    grappler_item_.reset(new GrapplerItem);
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii,
                                                &grappler_item_->graph));
    grappler_item_->id = "test_graph";
    grappler_item_->fetch = {"while/Exit", "while/Exit_1"};
  }

  void CreateGrapplerItemWithInterDeviceTransfers() {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(kCPU0);

    // Create a FusedBatchNorm op that has multiple output ports.
    auto x = ops::RandomUniform(
        s.WithOpName("x"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto scale =
        ops::RandomUniform(s.WithOpName("scale"), {depth_in_}, DT_FLOAT);
    auto offset =
        ops::RandomUniform(s.WithOpName("offset"), {depth_in_}, DT_FLOAT);
    auto mean = ops::RandomUniform(s.WithOpName("mean"), {0}, DT_FLOAT);
    auto var = ops::RandomUniform(s.WithOpName("var"), {0}, DT_FLOAT);

    auto batch_norm = ops::FusedBatchNorm(
        s.WithOpName("bn"), x, scale, offset, mean, var,
        ops::FusedBatchNorm::IsTraining(true).Epsilon(0.1f));
    auto y = batch_norm.y;
    auto batch_mean = batch_norm.batch_mean;
    auto batch_var = batch_norm.batch_variance;
    // y1 and y2 take the same tensor, so there should be only 1 Send and Recv.
    auto y1 = ops::Identity(s.WithOpName("y1").WithDevice(kCPU1), y);
    auto y2 = ops::Identity(s.WithOpName("y2").WithDevice(kCPU1), y);
    // batch_mean1 and batch_var1 take different output ports, so each will
    // initiate Send/Recv.
    auto batch_mean1 = ops::Identity(
        s.WithOpName("batch_mean1").WithDevice(kCPU1), batch_mean);
    auto batch_var1 =
        ops::Identity(s.WithOpName("batch_var1").WithDevice(kCPU1), batch_var);
    // This is control dependency.
    auto control_dep = ops::NoOp(s.WithOpName("control_dep")
                                     .WithControlDependencies(y)
                                     .WithDevice(kCPU1));

    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_conv2d_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"y1", "y2", "batch_mean1", "batch_var1",
                             "control_dep"};

    dependency_["bn"] = {"x", "mean", "var"};
    dependency_["y1"] = {"bn"};
    dependency_["y2"] = {"bn"};
    dependency_["batch_mean1"] = {"bn"};
    dependency_["batch_var1"] = {"bn"};
    dependency_["control_dep"] = {"bn"};
  }

  // Call this after creating grappler_item_ and setting up dependency_.
  void InitScheduler() {
    scheduler_.reset(new TestVirtualScheduler(
        grappler_item_.get(), true /* use_static_shapes */, cluster_.get()));
    TF_CHECK_OK(scheduler_->Init());
  }

  // Returns cost based on op.
  Costs SimplePredictCosts(const OpContext& op_context) const {
    Costs c;
    int64 exec_cost = 0;
    if (op_context.op_info.op() == "MatMul") {
      exec_cost = 2000000000;
    } else if (op_context.op_info.op() == "RandomUniform") {
      exec_cost = 1000000000;
    } else {
      exec_cost = 1000;
    }
    c.execution_time = Costs::NanoSeconds(exec_cost);
    return c;
  }

  // Call this after init scheduler_. Scheduler stops after executing
  // target_node.
  std::unordered_map<string, OpContext> RunScheduler(
      const string& target_node) {
    Costs zero_costs = Costs::ZeroCosts();
    std::unordered_map<string, OpContext> ops_executed;
    bool more_nodes = true;
    do {
      OpContext op_context = scheduler_->GetCurrNode();
      ops_executed[op_context.name] = op_context;
      std::cout << op_context.name << std::endl;

      Costs node_costs = SimplePredictCosts(op_context);

      // Check scheduling order.
      auto it = dependency_.find(op_context.name);
      if (it != dependency_.end()) {
        for (const auto& preceding_node : it->second) {
          EXPECT_GT(ops_executed.count(preceding_node), 0);
        }
      }
      more_nodes = scheduler_->MarkCurrNodeExecuted(node_costs);

      if (op_context.name == target_node) {
        // Scheduler has the state after executing the target node.
        break;
      }
    } while (more_nodes);
    return ops_executed;
  }

  // Helper method for validating a vector.
  template <typename T>
  void ExpectVectorEq(const std::vector<T>& expected,
                      const std::vector<T>& test_elements) {
    // Set of expected elements for an easy comparison.
    std::set<T> expected_set(expected.begin(), expected.end());
    for (const auto& element : test_elements) {
      EXPECT_GT(expected_set.count(element), 0);
    }
    EXPECT_EQ(expected.size(), test_elements.size());
  }

  // Helper method that checks the name of nodes.
  void ValidateNodeDefs(const std::vector<string>& expected,
                        const std::vector<const NodeDef*>& node_defs) {
    std::vector<string> node_names;
    std::transform(node_defs.begin(), node_defs.end(),
                   std::back_inserter(node_names),
                   [](const NodeDef* node) { return node->name(); });
    ExpectVectorEq(expected, node_names);
  }

  // Helper method for validating a set.
  template <typename T>
  void ExpectSetEq(const std::set<T>& expected,
                   const std::set<T>& test_elements) {
    for (const auto& element : test_elements) {
      EXPECT_GT(expected.count(element), 0);
    }
    EXPECT_EQ(expected.size(), test_elements.size());
  }

  // Helper method tthat checks name - port pairs.
  void ValidateMemoryUsageSnapshot(
      const std::vector<string>& expected_names, const int port_num_expected,
      const std::unordered_set<std::pair<const NodeDef*, int>,
                               DeviceState::NodePairHash>& mem_usage_snapshot) {
    std::set<std::pair<string, int>> nodes_at_peak_mem_usage;
    std::transform(
        mem_usage_snapshot.begin(), mem_usage_snapshot.end(),
        std::inserter(nodes_at_peak_mem_usage, nodes_at_peak_mem_usage.begin()),
        [](const std::pair<const NodeDef*, int>& node_port) {
          return std::make_pair(node_port.first->name(), node_port.second);
        });
    std::set<std::pair<string, int>> expected;
    std::transform(expected_names.begin(), expected_names.end(),
                   std::inserter(expected, expected.begin()),
                   [port_num_expected](const string& name) {
                     return std::make_pair(name, port_num_expected);
                   });
    ExpectSetEq(expected, nodes_at_peak_mem_usage);
  }

  // Helper method for checking nodes dependency.
  void ValidateDependencyChain(
      const std::unordered_map<string, int64>& start_times,
      const std::vector<string>& nodes_in_dependency_order) {
    int64 prev_node_time = -1;
    for (const auto& node : nodes_in_dependency_order) {
      int64 curr_node_time = start_times.at(node);
      EXPECT_GE(curr_node_time, prev_node_time);
      prev_node_time = curr_node_time;
    }
  }

  // Helper method for converting shape vector to TensorProperty.
  OpInfo::TensorProperties ShapeToTensorProperty(
      const std::vector<int> shape, const DataType& data_type) const {
    OpInfo::TensorProperties tensor_property;
    tensor_property.set_dtype(data_type);
    for (const auto& x : shape) {
      tensor_property.mutable_shape()->add_dim()->set_size(x);
    }
    return tensor_property;
  }

  // SetUp() inits cluster_ and placer_.
  std::unique_ptr<VirtualCluster> cluster_;
  std::unique_ptr<VirtualPlacer> placer_;

  // grappler_item_ and scheduler_ will be initialized differently for each test
  // case.
  std::unique_ptr<GrapplerItem> grappler_item_;
  std::unique_ptr<TestVirtualScheduler> scheduler_;
  // Node name -> its preceding nodes map for testing scheduling order.
  std::unordered_map<string, std::vector<string>> dependency_;

  // Shared params for Conv2D related graphs:
  const int batch_size_ = 4;
  const int width_ = 10;
  const int height_ = 10;
  const int depth_in_ = 8;
  const int kernel_ = 3;
  const int depth_out_ = 16;
};

// Test that FIFOManager correctly returns the current node with only 1 node.
TEST_F(VirtualSchedulerTest, GetSingleNodeFIFOManager) {
  // Init.
  FIFOManager manager = FIFOManager();

  // Add the node to FIFOManager.
  manager.AddNode(&node1_);
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
}

// Test that FIFOManager removes the only node contained within.
TEST_F(VirtualSchedulerTest, RemoveSingleNodeFIFOManager) {
  // Init.
  FIFOManager manager = FIFOManager();

  // Add the node to FIFOManager.
  manager.AddNode(&node1_);

  // Remove the only node in FIFOManager.
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

// Test that FIFOManager can remove multiple nodes and returns the current node
// in the right order
TEST_F(VirtualSchedulerTest, GetAndRemoveMultipleFIFOManager) {
  // Init.
  FIFOManager manager = FIFOManager();

  // Add the nodes to FIFOManager.
  manager.AddNode(&node1_);
  manager.AddNode(&node2_);
  manager.AddNode(&node3_);
  manager.AddNode(&node4_);

  // Keep checking current node while removing nodes from manager.
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

// Test that FIFOManager can remove multiple nodes and add more nodes, still
// returning the current node in the right order
TEST_F(VirtualSchedulerTest, AddAndRemoveMultipleFIFOManager) {
  // Init.
  FIFOManager manager = FIFOManager();

  // Add the nodes to FIFOManager.
  manager.AddNode(&node1_);
  manager.AddNode(&node2_);
  manager.AddNode(&node3_);
  manager.AddNode(&node4_);

  // Keep checking current node as nodes are removed and added.
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.AddNode(&node5_);
  // GetCurrNode()  should return the same node even if some nodes are added,
  // until RemoveCurrNode() is called.
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.AddNode(&node6_);
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node5", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

// Test that LIFOManager correctly returns the current node with only 1 node.
TEST_F(VirtualSchedulerTest, GetSingleNodeLIFOManager) {
  // Init.
  LIFOManager manager = LIFOManager();

  // Add the node to LIFOManager.
  manager.AddNode(&node1_);
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
}

// Test that LIFOManager removes the only node contained within.
TEST_F(VirtualSchedulerTest, RemoveSingleNodeLIFOManager) {
  // Init.
  LIFOManager manager = LIFOManager();

  // Add the node to LIFOManager.
  manager.AddNode(&node1_);

  // Remove the only node in LIFOManager.
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

// Test that LIFOManager can remove multiple nodes and returns the current node
// in the right order
TEST_F(VirtualSchedulerTest, GetAndRemoveMultipleLIFOManager) {
  // Init.
  LIFOManager manager = LIFOManager();

  // Add the nodes to LIFOManager.
  manager.AddNode(&node1_);
  manager.AddNode(&node2_);
  manager.AddNode(&node3_);
  manager.AddNode(&node4_);

  // Keep checking current node while removing nodes from manager.
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

// Test that LIFOManager can remove multiple nodes (must be removing the current
// node) and add more nodes, still returning the current node in the right order
TEST_F(VirtualSchedulerTest, AddAndRemoveMultipleLIFOManager) {
  // Init.
  LIFOManager manager = LIFOManager();

  // Add the nodes to LIFOManager.
  manager.AddNode(&node1_);
  manager.AddNode(&node2_);
  manager.AddNode(&node3_);
  manager.AddNode(&node4_);

  // Keep checking current node as nodes are removed and added.
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.AddNode(&node5_);
  // GetCurrNode()  should return the same node even if some nodes are added,
  // until RemoveCurrNode() is called.
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node5", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.AddNode(&node6_);
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, GetSingleNodeFirstReadyManager) {
  FirstReadyManager manager;
  manager.Init(&node_states_);

  manager.AddNode(&node1_);
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
}

TEST_F(VirtualSchedulerTest, RemoveSingleNodeFirstReadyManager) {
  FirstReadyManager manager;
  manager.Init(&node_states_);
  manager.AddNode(&node1_);
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, GetAndRemoveMultipleFirstReadyManager) {
  FirstReadyManager manager;
  manager.Init(&node_states_);
  // Insert nodes in some random order.
  manager.AddNode(&node2_);
  manager.AddNode(&node1_);
  manager.AddNode(&node4_);
  manager.AddNode(&node5_);
  manager.AddNode(&node3_);
  manager.AddNode(&node6_);

  // In whatever order we insert nodes, we get the same order based on nodes'
  // time_ready.
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node5", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, GetCurrNodeFirstReadyManager) {
  FirstReadyManager manager;
  manager.Init(&node_states_);
  // Insert nodes in some random order.
  manager.AddNode(&node2_);
  manager.AddNode(&node1_);
  manager.AddNode(&node4_);
  manager.AddNode(&node5_);
  manager.AddNode(&node3_);
  manager.AddNode(&node6_);

  // Among these nodes, node6 has the smallest time_ready, hence, GetCurrNode()
  // should return it.
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());
  // Now insret a few other nodes, but their time_ready's are even smaller than
  // that of Node6. Before calling RemoveCurrNode(), GetCurrNode() should return
  // the same node, Node6, in this case.

  NodeDef node7;
  NodeDef node8;
  NodeDef node9;
  NodeSetUp("Node7", kConv2D, kCPU0, 5, &node7);
  NodeSetUp("Node8", kConv2D, kCPU0, 4, &node8);
  NodeSetUp("Node9", kConv2D, kCPU0, 3, &node9);

  manager.AddNode(&node7);
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());

  manager.AddNode(&node8);
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());

  manager.RemoveCurrNode();
  // Now Node6 is removed, and GetCurrNode() will return Node8.
  EXPECT_EQ("Node8", manager.GetCurrNode()->name());

  // Again, AddNode shouldn't change GetCurrNode().
  manager.AddNode(&node9);
  EXPECT_EQ("Node8", manager.GetCurrNode()->name());

  manager.RemoveCurrNode();
  EXPECT_EQ("Node9", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node7", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node5", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, DeterminismInFirstReadyManager) {
  FirstReadyManager manager1;
  manager1.Init(&node_states_);
  FirstReadyManager manager2;
  manager2.Init(&node_states_);

  // 6 nodes with same time_ready.
  NodeDef node7;
  NodeDef node8;
  NodeDef node9;
  NodeDef node10;
  NodeDef node11;
  NodeDef node12;
  NodeSetUp("Node7", kConv2D, kCPU0, 1000, &node7);
  NodeSetUp("Node8", kConv2D, kCPU0, 1000, &node8);
  NodeSetUp("Node9", kConv2D, kCPU0, 1000, &node9);
  NodeSetUp("Node10", kConv2D, kCPU0, 1000, &node10);
  NodeSetUp("Node11", kConv2D, kCPU0, 1000, &node11);
  NodeSetUp("Node12", kConv2D, kCPU0, 1000, &node12);

  // Add the above 6 nodes to manager1.
  manager1.AddNode(&node7);
  manager1.AddNode(&node8);
  manager1.AddNode(&node9);
  manager1.AddNode(&node10);
  manager1.AddNode(&node11);
  manager1.AddNode(&node12);

  // Add the above 6 nodes to manager2, but in a different order.
  manager2.AddNode(&node8);
  manager2.AddNode(&node11);
  manager2.AddNode(&node9);
  manager2.AddNode(&node10);
  manager2.AddNode(&node7);
  manager2.AddNode(&node12);

  // Expect both managers return the same nodes for deterministic node
  // scheduling.
  EXPECT_EQ(manager1.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager1.RemoveCurrNode();
  manager2.RemoveCurrNode();

  EXPECT_EQ(manager1.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager1.RemoveCurrNode();
  manager2.RemoveCurrNode();

  EXPECT_EQ(manager1.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager1.RemoveCurrNode();
  manager2.RemoveCurrNode();

  EXPECT_EQ(manager1.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager1.RemoveCurrNode();
  manager2.RemoveCurrNode();

  EXPECT_EQ(manager1.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager1.RemoveCurrNode();
  manager2.RemoveCurrNode();

  EXPECT_EQ(manager1.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager1.RemoveCurrNode();
  manager2.RemoveCurrNode();

  EXPECT_TRUE(manager1.Empty());
  EXPECT_TRUE(manager2.Empty());
}

TEST_F(VirtualSchedulerTest, RemoveSingleNodeCompositeNodeManager) {
  CompositeNodeManager manager;
  manager.Init(&node_states_);
  manager.AddNode(&node1_);
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, RemoveSingleNodeComopsiteNodeManager) {
  CompositeNodeManager manager;
  manager.Init(&node_states_);

  manager.AddNode(&node1_);
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, GetAndRemoveMultipleComopsiteNodeManager) {
  CompositeNodeManager manager;
  manager.Init(&node_states_);

  // Add the nodes to LIFOManager.
  manager.AddNode(&node1_);
  manager.AddNode(&node2_);
  manager.AddNode(&node3_);
  manager.AddNode(&node4_);

  // Keep checking current node as nodes are removed and added.
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.AddNode(&node5_);
  // GetCurrNode()  should return the same node even if some nodes are added,
  // until RemoveCurrNode() is called.
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node5", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.AddNode(&node6_);
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, MultiDeviceSendRecvComopsiteNodeManager) {
  CompositeNodeManager manager;
  manager.Init(&node_states_);
  // Additional nodes on kCPU1
  NodeDef node7;
  NodeDef node8;
  NodeDef node9;
  NodeSetUp("Node7", kConv2D, kCPU1, 1001, &node7);
  NodeSetUp("Node8", kConv2D, kCPU1, 2001, &node8);
  NodeSetUp("Node9", kConv2D, kCPU1, 3001, &node9);

  // Send and Recv nodes.
  NodeDef send1;
  NodeDef send2;
  NodeDef recv1;
  NodeDef recv2;
  NodeSetUp("Send1", kSend, kChannelFrom0To1, 2002, &send1);
  NodeSetUp("Send2", kSend, kChannelFrom1To0, 2005, &send2);
  NodeSetUp("Recv1", kRecv, kCPU0, 2003, &recv1);
  NodeSetUp("Recv2", kRecv, kCPU1, 2004, &recv2);

  // Insert nodes.
  manager.AddNode(&node1_);
  manager.AddNode(&node2_);
  manager.AddNode(&node3_);
  manager.AddNode(&node4_);
  manager.AddNode(&node5_);
  manager.AddNode(&node6_);
  manager.AddNode(&node7);
  manager.AddNode(&node8);
  manager.AddNode(&node9);
  manager.AddNode(&send1);
  manager.AddNode(&send2);
  manager.AddNode(&recv1);
  manager.AddNode(&recv2);

  // on kCPU0; last one is node6_, on kCPU1: last one is node9;
  // so choose one that has earliest time_ready among node6_, node9,
  // Send1, Send2, Recv1, and Recv2.
  EXPECT_EQ("Node6", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Then, the next one on kCPU0 is node5_; choose the earliest time_ready node
  // among node5_, node9, Send1, Send2, Recv1, and Recv2.
  EXPECT_EQ("Node5", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose among node4_, node9, Send1, Send2, Recv1, and Recv2.
  EXPECT_EQ("Send1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose among node4_, node9, Sen2, Recv1, and Recv2.
  EXPECT_EQ("Recv1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose among node4_, node9, Send2, and Recv2.
  EXPECT_EQ("Recv2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose among node4_, node9, and Send2.
  EXPECT_EQ("Send2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose between node4_, node9.
  EXPECT_EQ("Node4", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose between node3_, node9.
  EXPECT_EQ("Node9", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose between node3_, node8.
  EXPECT_EQ("Node8", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Next, choose between node3_, node7.
  EXPECT_EQ("Node7", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  // Then, just the nodes on kCPU1 -- LIFO.
  EXPECT_EQ("Node3", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node2", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node1", manager.GetCurrNode()->name());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

TEST_F(VirtualSchedulerTest, DeterminismInCompositeNodeManager) {
  CompositeNodeManager manager;
  manager.Init(&node_states_);
  CompositeNodeManager manager2;
  manager2.Init(&node_states_);

  // 6 nodes with same time_ready.
  NodeDef node7;
  NodeDef node8;
  NodeDef node9;
  NodeDef node10;
  NodeDef node11;
  NodeDef node12;
  NodeSetUp("Node7", kConv2D, kCPU0, 1000, &node7);
  NodeSetUp("Node8", kSend, kCPU0, 1000, &node8);
  NodeSetUp("Node9", kRecv, kCPU0, 1000, &node9);
  NodeSetUp("Node10", kConv2D, kCPU0, 999, &node10);
  NodeSetUp("Node11", kRecv, kCPU0, 999, &node11);
  NodeSetUp("Node12", kConv2D, kCPU1, 1000, &node12);

  // Add Nodes 7 to 9 to manager.
  manager.AddNode(&node7);
  manager.AddNode(&node8);
  manager.AddNode(&node9);

  // It should return _Send, Recv, and the other op order, when the candidate
  // nodes have same time_ready.
  EXPECT_EQ("Node8", manager.GetCurrNode()->name());
  EXPECT_EQ(kSend, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node9", manager.GetCurrNode()->name());
  EXPECT_EQ(kRecv, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node7", manager.GetCurrNode()->name());
  EXPECT_EQ(kConv2D, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());

  // Add Nodes 7 to 9 to manager, but in a different order.
  manager.AddNode(&node9);
  manager.AddNode(&node8);
  manager.AddNode(&node7);

  // Expect same order (_Send, _Recv, and the other op), regardless of Add
  // order.
  EXPECT_EQ("Node8", manager.GetCurrNode()->name());
  EXPECT_EQ(kSend, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node9", manager.GetCurrNode()->name());
  EXPECT_EQ(kRecv, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node7", manager.GetCurrNode()->name());
  EXPECT_EQ(kConv2D, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());

  // Conv2D's time_ready < Send's time_ready; Expect Conv2D first.
  manager.AddNode(&node8);
  manager.AddNode(&node10);
  EXPECT_EQ("Node10", manager.GetCurrNode()->name());
  EXPECT_EQ(kConv2D, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node8", manager.GetCurrNode()->name());
  EXPECT_EQ(kSend, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());

  // Recv's time_ready < Send' time_ready; Expect Recv first.
  manager.AddNode(&node11);
  manager.AddNode(&node8);
  EXPECT_EQ("Node11", manager.GetCurrNode()->name());
  EXPECT_EQ(kRecv, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_EQ("Node8", manager.GetCurrNode()->name());
  EXPECT_EQ(kSend, manager.GetCurrNode()->op());
  manager.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());

  // Node7 and 12 are normal ops with the same time_ready, placed on different
  // devices. These two nodes are added to manager and manager2, but in
  // different orders; Expect GetCurrNode() returns the nodes in the same order.
  manager.AddNode(&node7);
  manager.AddNode(&node12);

  manager2.AddNode(&node12);
  manager2.AddNode(&node7);

  EXPECT_EQ(manager.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager.RemoveCurrNode();
  manager2.RemoveCurrNode();
  EXPECT_EQ(manager.GetCurrNode()->name(), manager2.GetCurrNode()->name());
  manager.RemoveCurrNode();
  manager2.RemoveCurrNode();
  EXPECT_TRUE(manager.Empty());
}

// Create small graph, run predict costs on it, make sure the costs from the
// summary match the hand-calculated costs.
TEST_F(VirtualSchedulerTest, SummaryCostTest) {
  // Run matmul test.
  CreateGrapplerItemWithMatmulChain();
  InitScheduler();
  auto ops_executed = RunScheduler("");
  Costs c = scheduler_->Summary();

  // RandomUniform - 5 * 1s
  // Matmuls - 4 * 2s = 8
  // Misc - 5 * 1us
  // Total: 13000005
  EXPECT_EQ(13000005, c.execution_time.asMicroSeconds().count());
}

// Like the above SummaryCostTest, but makes sure the stepstats timeline is
// correct.
TEST_F(VirtualSchedulerTest, SummaryCostStepStatsTest) {
  // Run matmul test.
  CreateGrapplerItemWithMatmulChain();
  InitScheduler();
  auto ops_executed = RunScheduler("");
  RunMetadata metadata;
  Costs c = scheduler_->Summary(&metadata);
  StepStats stepstats = metadata.step_stats();
  EXPECT_EQ(13000005, c.execution_time.asMicroSeconds().count());

  // Should only be 1 device!
  EXPECT_EQ(1, stepstats.dev_stats().size());

  // Create a map of op name -> start and end times (micros).
  std::map<string, std::pair<int64, int64>> start_end_times;
  for (const auto& device_step_stats : stepstats.dev_stats()) {
    for (const auto& stats : device_step_stats.node_stats()) {
      int64 start = stats.all_start_micros();
      int64 end = start + stats.all_end_rel_micros();
      start_end_times[stats.node_name()] = std::pair<int64, int64>(start, end);

      // Make sure that the output properties are correct for
      // MatMul and RandomUniform operations.
      // We only check for dtype, and shape (excluding alloc)
      // since alloc is not set by the virtual scheduler.
      if (stats.timeline_label() == "MatMul" ||
          stats.timeline_label() == "RandomUniform") {
        EXPECT_EQ(1, stats.output().size());
        for (const auto& output : stats.output()) {
          EXPECT_EQ(DT_FLOAT, output.tensor_description().dtype());
          EXPECT_EQ(2, output.tensor_description().shape().dim().size());
          for (const auto& dim : output.tensor_description().shape().dim()) {
            EXPECT_EQ(3200, dim.size());
          }
        }
      }
    }
  }

  // The base start_time is the time to compute RandomUniforms
  int64 cur_time = static_cast<int64>(5000005);
  // The increment is the execution time of one matmul. See
  // CreateGrapplerItemWithMatmulChain for details.
  int64 increment = static_cast<int64>(2000000);
  auto op_names = {"ab", "abc", "abcd", "abcde"};
  for (const auto& op_name : op_names) {
    int64 actual_start = start_end_times[op_name].first;
    int64 actual_end = start_end_times[op_name].second;
    int64 expected_start = cur_time;
    int64 expected_end = cur_time + increment;
    EXPECT_EQ(expected_start, actual_start);
    EXPECT_EQ(expected_end, actual_end);
    cur_time += increment;
  }
}

TEST_F(VirtualSchedulerTest, InitAndBasicScheduling) {
  // Init.
  CreateGrapplerItemWithConv2Ds();
  InitScheduler();

  // Run the scheduler.
  auto ops_executed = RunScheduler("");  // Run all the nodes.

  // [const and rand] * (x, y, f), and c0 and c1. c2 and z shouldn't be
  // executed.
  EXPECT_EQ(8, ops_executed.size());

  // x, y, f, c0, and c1 should be in the ops executed.
  EXPECT_GT(ops_executed.count("x"), 0);
  EXPECT_GT(ops_executed.count("y"), 0);
  EXPECT_GT(ops_executed.count("f"), 0);
  EXPECT_GT(ops_executed.count("c0"), 0);
  EXPECT_GT(ops_executed.count("c1"), 0);

  // z and c2 shouldn't be part of it.
  EXPECT_EQ(ops_executed.count("z"), 0);
  EXPECT_EQ(ops_executed.count("c2"), 0);

  // Check input / output properties.
  EXPECT_EQ(1, ops_executed["x"].op_info.outputs_size());
  EXPECT_EQ(1, ops_executed["y"].op_info.outputs_size());
  EXPECT_EQ(1, ops_executed["f"].op_info.outputs_size());
  EXPECT_EQ(2, ops_executed["c0"].op_info.inputs_size());
  EXPECT_EQ(2, ops_executed["c1"].op_info.inputs_size());
}

TEST_F(VirtualSchedulerTest, CalculateOutputSize) {
  // Init.
  CreateGrapplerItemWithAddN();
  InitScheduler();

  // Create a set of tensor properties.
  std::vector<OpInfo::TensorProperties> output;
  output.push_back(ShapeToTensorProperty({4, 4}, DT_FLOAT));           // 0
  output.push_back(ShapeToTensorProperty({1}, DT_FLOAT));              // 1
  output.push_back(ShapeToTensorProperty({10, 10, 10}, DT_HALF));      // 2
  output.push_back(ShapeToTensorProperty({100, 7, 8, 99}, DT_FLOAT));  // 3
  output.push_back(ShapeToTensorProperty({-1, 7, 8, 99}, DT_FLOAT));   // 4
  output.push_back(ShapeToTensorProperty({-1, 7, -1, 99}, DT_FLOAT));  // 4

  // port_num -1 is for control dependency: hard coded 4B.
  EXPECT_EQ(4, scheduler_->CalculateOutputSize(output, -1));

  // Test valid outputs.
  EXPECT_EQ(4 * 4 * 4, scheduler_->CalculateOutputSize(output, 0));
  EXPECT_EQ(4 * 1, scheduler_->CalculateOutputSize(output, 1));
  EXPECT_EQ(2 * 10 * 10 * 10, scheduler_->CalculateOutputSize(output, 2));
  EXPECT_EQ(4 * 100 * 7 * 8 * 99, scheduler_->CalculateOutputSize(output, 3));

  // Any unknown shape (-1) shall yield zero output size.
  EXPECT_EQ(0, scheduler_->CalculateOutputSize(output, 4));
  EXPECT_EQ(0, scheduler_->CalculateOutputSize(output, 5));

  // Invalid port_num (though it may be an error) shall yield zero
  // output size.
  EXPECT_EQ(0, scheduler_->CalculateOutputSize(output, 6));
}

TEST_F(VirtualSchedulerTest, MemoryUsage) {
  // Init.
  CreateGrapplerItemWithAddN();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("");

  const auto* device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states->at(kCPU0);

  // out node adds 4 tensors, each with 10x10x10x10, so the peak memory usage
  // is 4 x the input tensor size while executing the out node.
  int64 one_input_node_size = 4 * 10 * 10 * 10 * 10;
  const std::vector<string> expected_names = {"x", "y", "z", "w"};
  EXPECT_EQ(expected_names.size() * one_input_node_size,
            cpu_state.max_memory_usage);
  ValidateMemoryUsageSnapshot(expected_names, 0 /* port_num_expected */,
                              cpu_state.mem_usage_snapshot_at_peak);
}

TEST_F(VirtualSchedulerTest, UnnecessaryFeedNodes) {
  CreateGrapplerItemWithUnnecessaryPlaceholderNodes();
  InitScheduler();

  // Test that scheduler can run graphs with extra unnecessary feed nodes.
  auto ops_executed = RunScheduler("");
  ASSERT_EQ(1, ops_executed.size());
  ASSERT_EQ(ops_executed.count("x"), 1);
}

TEST_F(VirtualSchedulerTest, ControlDependency) {
  // Init.
  CreateGrapplerItemWithControlDependency();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("");

  const auto* device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states->at(kCPU0);

  // The graph has a NoOp that takes control dependency from 7 NoOps. The peak
  // memory usage is when executing the final NoOp.
  int64 one_input_node_size = 4;  // control dependency
  const std::vector<string> expected_names = {"x", "y", "z", "w",
                                              "u", "v", "t"};
  EXPECT_EQ(expected_names.size() * one_input_node_size,
            cpu_state.max_memory_usage);
  ValidateMemoryUsageSnapshot(expected_names, -1 /* port_num_expected */,
                              cpu_state.mem_usage_snapshot_at_peak);
}

TEST_F(VirtualSchedulerTest, ComplexDependency) {
  // Init.
  CreateGrapplerItemWithBatchNorm();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("bn");

  const auto& device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states->at(kCPU0);

  // The graph is
  //  bn = FusedBatchNorm(x, scale, offset, mean, var)
  //  z1 = bn.y + x
  //  z2 = bn.var + bn.var
  //  z3 = bn.var + bn.var
  //  z4 = control dependency from bn.
  //  Note that bn.mean doesn't have any consumer.
  const int x_size = batch_size_ * width_ * height_ * depth_in_;
  int64 expected_size =
      4 * (2 * x_size /* x and bn.y */ + depth_in_ /* bn.var */ +
           1 /* control dependency */);
  EXPECT_EQ(expected_size, cpu_state.memory_usage);

  // Nodes currently in memory: bn's port -1, 0, and 2, and x's port 0.
  std::set<std::pair<string, int>> nodes_in_memory;
  std::transform(
      cpu_state.nodes_in_memory.begin(), cpu_state.nodes_in_memory.end(),
      std::inserter(nodes_in_memory, nodes_in_memory.begin()),
      [](const std::pair<const NodeDef*, int>& node_port) {
        return std::make_pair(node_port.first->name(), node_port.second);
      });
  std::set<std::pair<string, int>> expected = {
      std::make_pair("bn", -1),
      std::make_pair("bn", 0),
      std::make_pair("bn", 2),
      std::make_pair("x", 0),
  };
  ExpectSetEq(expected, nodes_in_memory);

  const auto* node_states = scheduler_->GetNodeStates();
  const NodeState* bn_node = nullptr;
  const NodeState* x_node = nullptr;
  for (const auto& nodedef_node_state : *node_states) {
    const NodeDef* node = nodedef_node_state.first;
    const NodeState& node_state = nodedef_node_state.second;
    if (node->name() == "bn") {
      bn_node = &node_state;
    }
    if (node->name() == "x") {
      x_node = &node_state;
    }
  }
  CHECK_NOTNULL(bn_node);
  CHECK_NOTNULL(x_node);

  ValidateNodeDefs({"bn", "z1"}, x_node->outputs.at(0));
  ValidateNodeDefs({"z4"}, bn_node->outputs.at(-1));
  ValidateNodeDefs({"z1"}, bn_node->outputs.at(0));
  // z2 and z3 are bn.var + bn.var, so they appear twice in bn's output port 2.
  ValidateNodeDefs({"z2", "z3", "z2", "z3"}, bn_node->outputs.at(2));
}

TEST_F(VirtualSchedulerTest, Variable) {
  // Init.
  CreateGrapplerItemWithConv2DAndVariable();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("");

  const auto* device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states->at(kCPU0);

  // There is one Conv2D that takes x and f, but f is variable, so it should be
  // in persistent nodes.
  // f is variable.
  ValidateMemoryUsageSnapshot({"f"}, 0 /* port_num_expected */,
                              cpu_state.persistent_nodes);
  // Only x in peak memory usage snapshot.
  ValidateMemoryUsageSnapshot({"x"}, 0 /* port_num_expected */,
                              cpu_state.mem_usage_snapshot_at_peak);
}

TEST_F(VirtualSchedulerTest, WhileLoop) {
  // Init.
  CreateGrapplerItemWithLoop();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("");

  // Check the timeline
  RunMetadata metadata;
  scheduler_->Summary(&metadata);

  // Nodes in topological order:
  // * const, ones
  // * while/Enter, while/Enter_1
  // * while/Merge, while/Merge_1
  // * while/Less/y
  // * while/Less
  // * while/LoopCond
  // * while/Switch, while/Switch_1
  // * while/Identity, while/Identity_1, while/Exit, while/Exit_1
  // * while/add/y, while/concat/axis
  // * while/add, while/concat
  // * while/NextIteration, while/NextIteration_1

  int num_next_iteration = 0;
  int num_next_iteration_1 = 0;
  int num_exit = 0;
  int num_exit_1 = 0;
  int64 next_iter_start_micro;
  int64 next_iter_1_start_micro;
  int64 exit_start_micro;
  int64 exit_1_start_micro;

  std::unordered_map<string, int64> start_times;
  for (const auto& device_step_stats : metadata.step_stats().dev_stats()) {
    for (const auto& stats : device_step_stats.node_stats()) {
      start_times[stats.node_name()] = stats.all_start_micros();
      if (stats.node_name() == "while/NextIteration") {
        ++num_next_iteration;
        next_iter_start_micro = stats.all_start_micros();
      } else if (stats.node_name() == "while/NextIteration_1") {
        ++num_next_iteration_1;
        next_iter_1_start_micro = stats.all_start_micros();
      } else if (stats.node_name() == "while/Exit") {
        ++num_exit;
        exit_start_micro = stats.all_start_micros();
      } else if (stats.node_name() == "while/Exit_1") {
        ++num_exit_1;
        exit_1_start_micro = stats.all_start_micros();
      }
    }
  }

  // Make sure we went though the body of the loop once, and that the output of
  // the loop was scheduled as well.
  EXPECT_EQ(1, num_next_iteration);
  EXPECT_EQ(1, num_next_iteration_1);
  EXPECT_EQ(1, num_exit);
  EXPECT_EQ(1, num_exit_1);

  // Start times of while/NextIteration and while/NextIteration_1 should be
  // different, so should be those of while/Exit and while/Exit_1.
  EXPECT_NE(next_iter_start_micro, next_iter_1_start_micro);
  EXPECT_NE(exit_start_micro, exit_1_start_micro);

  // Check dependency among the nodes; no matter what scheduling mechanism we
  // use, the scheduled ops should follow these dependency chains.
  // Note that currently, VirtualScheduler executes while/Merge twice; hence,
  // we're not testing dependency chains related to while/Merge.
  // TODO(dyoon): after fixing while loop behavior correctly (run nodes in the
  // order of Enter, Merge, ...loop condition ..., ... loop body ...,
  // NextIteration, Merge, ... loop condition ..., Exit), re-enable dependency
  // chaing test w/ Merge nodes.
  ValidateDependencyChain(
      start_times,
      {"Const", "while/Enter",  // "while/Merge",
       "while/Less/y", "while/Less", "while/LoopCond", "while/Switch",
       "while/Identity", "while/add/y", "while/add", "while/NextIteration"});
  // ValidateDependencyChain(start_times, {"while/Merge", "while/Less"});
  ValidateDependencyChain(start_times,
                          {"ones", "while/Enter_1",  // "while/Merge_1",
                           "while/Switch_1", "while/Identity_1", "while/concat",
                           "while/NextIteration_1"});
  ValidateDependencyChain(start_times, {"while/Switch", "while/Exit"});
  ValidateDependencyChain(
      start_times, {"while/Identity", "while/concat/axis", "while/concat"});
  ValidateDependencyChain(start_times, {"while/Identity", "while/add"});
  ValidateDependencyChain(start_times, {"while/Switch_1", "while/Exit_1"});
}

TEST_F(VirtualSchedulerTest, InterDeviceTransfer) {
  // Init.
  CreateGrapplerItemWithInterDeviceTransfers();
  InitScheduler();

  // Run the scheduler.
  auto ops_executed = RunScheduler("");

  // Helper lambda to extract port num from _Send and _Recv op name.
  auto get_port_num = [](const string& name) -> int {
    if (name.find("bn_0") != std::string::npos) {
      return 0;
    } else if (name.find("bn_1") != std::string::npos) {
      return 1;
    } else if (name.find("bn_2") != std::string::npos) {
      return 2;
    } else if (name.find("bn_minus1") != std::string::npos) {
      return -1;
    }
    return -999;
  };

  // Reorganize ops_executed for further testing.
  std::unordered_map<string, int> op_count;
  std::unordered_map<int, string> recv_op_names;
  std::unordered_map<int, string> send_op_names;
  for (const auto& x : ops_executed) {
    const auto& name = x.first;
    const auto& node_info = x.second;
    const auto& op = node_info.op_info.op();
    if (op == kRecv) {
      recv_op_names[get_port_num(name)] = name;
    } else if (op == kSend) {
      send_op_names[get_port_num(name)] = name;
    }
    op_count[op]++;
  }

  // Same number of _Send and _Recv.
  EXPECT_EQ(op_count.at(kSend), op_count.at(kRecv));

  // Expect 4 Send and Recvs each: port 0, 1, and, 2, and control dependency.
  EXPECT_EQ(op_count.at(kRecv), 4);
  EXPECT_EQ(op_count.at(kSend), 4);

  // Helper lambda for extracting output Tensor size.
  auto get_output_size = [this, ops_executed](const string& name) -> int64 {
    const auto& output_properties_ = ops_executed.at(name).op_info.outputs();
    std::vector<OpInfo::TensorProperties> output_properties;
    for (const auto& output_property : output_properties_) {
      output_properties.push_back(output_property);
    }
    return scheduler_->CalculateOutputSize(output_properties, 0);
  };

  // Validate transfer size.
  // Batchnorm output y is 4D vector: batch x width x width x depth.
  int input_size = 4 * batch_size_ * width_ * height_ * depth_in_;
  EXPECT_EQ(get_output_size(recv_op_names[0]), input_size);
  EXPECT_EQ(get_output_size(send_op_names[0]), input_size);
  // Mean and vars are 1-D vector with size depth_in_.
  EXPECT_EQ(get_output_size(recv_op_names[1]), 4 * depth_in_);
  EXPECT_EQ(get_output_size(send_op_names[1]), 4 * depth_in_);
  EXPECT_EQ(get_output_size(recv_op_names[2]), 4 * depth_in_);
  EXPECT_EQ(get_output_size(send_op_names[2]), 4 * depth_in_);
  // Control dependency size is 4B.
  EXPECT_EQ(get_output_size(recv_op_names[-1]), 4);
  EXPECT_EQ(get_output_size(send_op_names[-1]), 4);
}

TEST_F(VirtualSchedulerTest, GraphWithSendRecv) {
  // Init.
  CreateGrapplerItemWithSendRecv();
  InitScheduler();

  // Run the scheduler.
  auto ops_executed = RunScheduler("");

  EXPECT_GT(ops_executed.count("Const"), 0);
  EXPECT_GT(ops_executed.count("Send"), 0);
  EXPECT_GT(ops_executed.count("Recv"), 0);
}

TEST_F(VirtualSchedulerTest, GraphWithSendRecvDifferentDevice) {
  // Init.
  CreateGrapplerItemWithSendRecv();
  // Change Recv node's device so that Send and Recv are placed on different
  // devices.
  auto& graph = grappler_item_->graph;
  const string recv_device = kCPU1;
  for (int i = 0; i < graph.node_size(); i++) {
    auto* node = graph.mutable_node(i);
    if (node->name() == "Recv") {
      node->set_device(recv_device);
      auto* attr = node->mutable_attr();
      (*attr)["recv_device"].set_s(recv_device);
    } else if (node->name() == "Send") {
      auto* attr = node->mutable_attr();
      (*attr)["recv_device"].set_s(recv_device);
    }
  }
  InitScheduler();

  // Run the scheduler.
  auto ops_executed = RunScheduler("");

  // Expect Const, Send, Recv, and VirtualScheduler created Send and Recv ops.
  EXPECT_GT(ops_executed.count("Const"), 0);
  EXPECT_GT(ops_executed.count("Send"), 0);
  EXPECT_GT(ops_executed.count("Send_Send_0_from_/job_localhost/replica_0/"
                               "task_0/cpu_0_to_/job_localhost"
                               "/replica_0/task_0/cpu_1"),
            0);
  EXPECT_GT(ops_executed.count(
                "Recv_Send_0_on_/job_localhost/replica_0/task_0/cpu_1"),
            0);
  EXPECT_GT(ops_executed.count("Recv"), 0);
}

TEST_F(VirtualSchedulerTest, GraphWihtOnlyRecv) {
  // Init.
  CreateGrapplerItemWithRecvWithoutSend();
  InitScheduler();

  // Run the scheduler.
  auto ops_executed = RunScheduler("");

  // Recv without Send will be treated as initially ready node.
  EXPECT_GT(ops_executed.count("Recv"), 0);
}
}  // end namespace grappler
}  // end namespace tensorflow
