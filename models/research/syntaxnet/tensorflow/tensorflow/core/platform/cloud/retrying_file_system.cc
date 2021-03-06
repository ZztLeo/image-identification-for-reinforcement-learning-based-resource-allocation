/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/cloud/retrying_file_system.h"
#include <functional>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/cloud/retrying_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

namespace {

class RetryingRandomAccessFile : public RandomAccessFile {
 public:
  RetryingRandomAccessFile(std::unique_ptr<RandomAccessFile> base_file,
                           int64 delay_microseconds)
      : base_file_(std::move(base_file)),
        initial_delay_microseconds_(delay_microseconds) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    return RetryingUtils::CallWithRetries(
        std::bind(&RandomAccessFile::Read, base_file_.get(), offset, n, result,
                  scratch),
        initial_delay_microseconds_);
  }

 private:
  std::unique_ptr<RandomAccessFile> base_file_;
  const int64 initial_delay_microseconds_;
};

class RetryingWritableFile : public WritableFile {
 public:
  RetryingWritableFile(std::unique_ptr<WritableFile> base_file,
                       int64 delay_microseconds)
      : base_file_(std::move(base_file)),
        initial_delay_microseconds_(delay_microseconds) {}

  ~RetryingWritableFile() override {
    // Makes sure the retrying version of Close() is called in the destructor.
    Close().IgnoreError();
  }

  Status Append(const StringPiece& data) override {
    return RetryingUtils::CallWithRetries(
        std::bind(&WritableFile::Append, base_file_.get(), data),
        initial_delay_microseconds_);
  }
  Status Close() override {
    return RetryingUtils::CallWithRetries(
        std::bind(&WritableFile::Close, base_file_.get()),
        initial_delay_microseconds_);
  }
  Status Flush() override {
    return RetryingUtils::CallWithRetries(
        std::bind(&WritableFile::Flush, base_file_.get()),
        initial_delay_microseconds_);
  }
  Status Sync() override {
    return RetryingUtils::CallWithRetries(
        std::bind(&WritableFile::Sync, base_file_.get()),
        initial_delay_microseconds_);
  }

 private:
  std::unique_ptr<WritableFile> base_file_;
  const int64 initial_delay_microseconds_;
};

}  // namespace

Status RetryingFileSystem::NewRandomAccessFile(
    const string& filename, std::unique_ptr<RandomAccessFile>* result) {
  std::unique_ptr<RandomAccessFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::NewRandomAccessFile, base_file_system_.get(),
                filename, &base_file),
      initial_delay_microseconds_));
  result->reset(new RetryingRandomAccessFile(std::move(base_file),
                                             initial_delay_microseconds_));
  return Status::OK();
}

Status RetryingFileSystem::NewWritableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::NewWritableFile, base_file_system_.get(), filename,
                &base_file),
      initial_delay_microseconds_));
  result->reset(new RetryingWritableFile(std::move(base_file),
                                         initial_delay_microseconds_));
  return Status::OK();
}

Status RetryingFileSystem::NewAppendableFile(
    const string& filename, std::unique_ptr<WritableFile>* result) {
  std::unique_ptr<WritableFile> base_file;
  TF_RETURN_IF_ERROR(RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::NewAppendableFile, base_file_system_.get(),
                filename, &base_file),
      initial_delay_microseconds_));
  result->reset(new RetryingWritableFile(std::move(base_file),
                                         initial_delay_microseconds_));
  return Status::OK();
}

Status RetryingFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& filename, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::NewReadOnlyMemoryRegionFromFile,
                base_file_system_.get(), filename, result),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::FileExists(const string& fname) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::FileExists, base_file_system_.get(), fname),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::Stat(const string& fname, FileStatistics* stat) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::Stat, base_file_system_.get(), fname, stat),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::GetChildren(const string& dir,
                                       std::vector<string>* result) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::GetChildren, base_file_system_.get(), dir, result),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::GetMatchingPaths(const string& pattern,
                                            std::vector<string>* result) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::GetMatchingPaths, base_file_system_.get(), pattern,
                result),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::DeleteFile(const string& fname) {
  return RetryingUtils::DeleteWithRetries(
      std::bind(&FileSystem::DeleteFile, base_file_system_.get(), fname),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::CreateDir(const string& dirname) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::CreateDir, base_file_system_.get(), dirname),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::DeleteDir(const string& dirname) {
  return RetryingUtils::DeleteWithRetries(
      std::bind(&FileSystem::DeleteDir, base_file_system_.get(), dirname),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::GetFileSize(const string& fname, uint64* file_size) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::GetFileSize, base_file_system_.get(), fname,
                file_size),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::RenameFile(const string& src, const string& target) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::RenameFile, base_file_system_.get(), src, target),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::IsDirectory(const string& dirname) {
  return RetryingUtils::CallWithRetries(
      std::bind(&FileSystem::IsDirectory, base_file_system_.get(), dirname),
      initial_delay_microseconds_);
}

Status RetryingFileSystem::DeleteRecursively(const string& dirname,
                                             int64* undeleted_files,
                                             int64* undeleted_dirs) {
  return RetryingUtils::DeleteWithRetries(
      std::bind(&FileSystem::DeleteRecursively, base_file_system_.get(),
                dirname, undeleted_files, undeleted_dirs),
      initial_delay_microseconds_);
}

void RetryingFileSystem::FlushCaches() { base_file_system_->FlushCaches(); }

}  // namespace tensorflow
