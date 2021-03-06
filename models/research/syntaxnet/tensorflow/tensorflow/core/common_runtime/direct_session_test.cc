/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

CallableOptions MakeCallableOptions(gtl::ArraySlice<string> feeds,
                                    gtl::ArraySlice<string> fetches,
                                    gtl::ArraySlice<string> targets) {
  CallableOptions ret;
  for (const string& feed : feeds) {
    ret.add_feed(feed);
  }
  for (const string& fetch : fetches) {
    ret.add_fetch(fetch);
  }
  for (const string& target : targets) {
    ret.add_target(target);
  }
  return ret;
}

std::unique_ptr<Session> CreateSession() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  return std::unique_ptr<Session>(NewSession(options));
}

class DirectSessionMinusAXTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    a_ = a->name();

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    Node* z = test::graph::Unary(&graph, "Identity", y_neg);
    z_ = z->name();
    z->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    test::graph::ToGraphDef(&graph, &def_);
  }

  string a_;
  string x_;
  string y_;
  string y_neg_;
  string z_;
  GraphDef def_;
};

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  TF_ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork_Callable) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Run the test twice to ensure that the Make/Run/Release cycle is hermetic.
  for (int i = 0; i < 2; ++i) {
    // Request two targets: one fetch output and one non-fetched output.
    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(
        MakeCallableOptions({}, {y_ + ":0"}, {y_neg_}), &handle));

    for (int i = 0; i < 2; ++i) {
      std::vector<Tensor> outputs;
      TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));

      ASSERT_EQ(1, outputs.size());
      // The first output should be initialized and have the correct
      // output.
      auto mat = outputs[0].matrix<float>();
      ASSERT_TRUE(outputs[0].IsInitialized());
      EXPECT_FLOAT_EQ(5.0, mat(0, 0));
    }

    Status s = session->RunCallable(handle, {}, nullptr, nullptr);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                      "`fetch_tensors` must be provided"));

    TF_ASSERT_OK(session->ReleaseCallable(handle));

    std::vector<Tensor> outputs;
    s = session->RunCallable(handle, {}, &outputs, nullptr);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(str_util::StrContains(
        s.error_message(),
        "Attempted to run callable after handle was released"));

    s = session->RunCallable(handle + 1, {}, &outputs, nullptr);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(
        str_util::StrContains(s.error_message(), "No such callable handle"));
  }
}

TEST_F(DirectSessionMinusAXTest, TestTensorConnection) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  {
    // Directly wire the output of node a to the output of node y, making the
    // callable graph into "Neg(a);".
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(y_neg_ + ":0");

    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
    ASSERT_EQ(1, outputs.size());
    auto mat = outputs[0].matrix<float>();
    ASSERT_TRUE(outputs[0].IsInitialized());
    EXPECT_FLOAT_EQ(-3.0, mat(0, 0));
    EXPECT_FLOAT_EQ(-2.0, mat(0, 1));
    EXPECT_FLOAT_EQ(1.0, mat(1, 0));
    EXPECT_FLOAT_EQ(0.0, mat(1, 1));
    TF_ASSERT_OK(session->ReleaseCallable(handle));
  }

  {
    // Directly wire the output of node a to the output of node y, making the
    // callable graph into "Neg(a);"; also fetch the result of a.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(a_ + ":0");
    callable_options.add_fetch(y_neg_ + ":0");

    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
    ASSERT_EQ(2, outputs.size());
    auto mat_a = outputs[0].matrix<float>();
    ASSERT_TRUE(outputs[0].IsInitialized());
    EXPECT_FLOAT_EQ(3.0, mat_a(0, 0));
    EXPECT_FLOAT_EQ(2.0, mat_a(0, 1));
    EXPECT_FLOAT_EQ(-1.0, mat_a(1, 0));
    EXPECT_FLOAT_EQ(0.0, mat_a(1, 1));

    auto mat_y_neg = outputs[1].matrix<float>();
    ASSERT_TRUE(outputs[1].IsInitialized());
    EXPECT_FLOAT_EQ(-3.0, mat_y_neg(0, 0));
    EXPECT_FLOAT_EQ(-2.0, mat_y_neg(0, 1));
    EXPECT_FLOAT_EQ(1.0, mat_y_neg(1, 0));
    EXPECT_FLOAT_EQ(0.0, mat_y_neg(1, 1));
    TF_ASSERT_OK(session->ReleaseCallable(handle));
  }

  {
    // Wire the output of "Neg(Matmul(a, x))" to the output of "a",
    // creating an invalid cycle.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(y_ + ":0");
    c->set_to_tensor(a_ + ":0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(
        str_util::StrContains(s.error_message(), "would create a cycle"));
  }

  {
    // Attempt to wire a non-existent node to a node that does exist.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor("unknown_node:0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(str_util::StrContains(s.error_message(), "unknown node"));
  }

  {
    // Attempt to wire a non-existent output from a node that does
    // exist to another node.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":17");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(str_util::StrContains(s.error_message(), "unknown edge"));
  }

  {
    // Attempt to wire a tensor to a node that doesn't exist.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor("unknown_node:0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsNotFound(s));
    EXPECT_TRUE(
        str_util::StrContains(s.error_message(), "unable to find feed output"));
  }

  {
    // Attempt to wire two tensors to the same tensor.
    CallableOptions callable_options;
    TensorConnection* c1 = callable_options.add_tensor_connection();
    c1->set_from_tensor(a_ + ":0");
    c1->set_to_tensor(y_neg_ + ":0");
    TensorConnection* c2 = callable_options.add_tensor_connection();
    c2->set_from_tensor(x_ + ":0");
    c2->set_to_tensor(y_neg_ + ":0");
    callable_options.add_fetch(z_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(str_util::StrContains(s.error_message(), "fed more than once"));
  }

  {
    // Attempt to wire a tensor to a tensor that is also being fed.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_feed(y_ + ":0");
    callable_options.add_fetch(y_neg_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(str_util::StrContains(s.error_message(), "fed more than once"));
  }
}

TEST_F(DirectSessionMinusAXTest, TestFeed) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  //
  // Note that the input being fed is on the second device.
  Tensor t(DT_FLOAT, TensorShape({2, 1}));
  t.matrix<float>()(0, 0) = 5;
  t.matrix<float>()(1, 0) = 6;
  std::vector<std::pair<string, Tensor>> inputs = {{x_, t}};
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<Tensor> outputs;

  // Run the graph
  Status s = session->Run(inputs, output_names, {}, &outputs);
  TF_ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  auto mat = outputs[0].matrix<float>();

  // Expect outputs to be; 1*5 + 2*6, 3*5 + 4*6
  EXPECT_FLOAT_EQ(17.0, mat(0, 0));
  EXPECT_FLOAT_EQ(39.0, mat(1, 0));
}

TEST_F(DirectSessionMinusAXTest, TestFeed_Callable) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  //
  // Note that the input being fed is on the second device.
  CallableOptions callable_options;
  callable_options.add_feed(x_);
  callable_options.add_fetch(y_ + ":0");
  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(MakeCallableOptions({x_}, {y_ + ":0"}, {}),
                                     &handle));
  Tensor t(DT_FLOAT, TensorShape({2, 1}));
  t.matrix<float>()(0, 0) = 5;
  t.matrix<float>()(1, 0) = 6;
  std::vector<Tensor> inputs = {t};
  std::vector<Tensor> outputs;

  // Run the callable
  TF_ASSERT_OK(session->RunCallable(handle, inputs, &outputs, nullptr));

  ASSERT_EQ(1, outputs.size());
  auto mat = outputs[0].matrix<float>();

  // Expect outputs to be; 1*5 + 2*6, 3*5 + 4*6
  EXPECT_FLOAT_EQ(17.0, mat(0, 0));
  EXPECT_FLOAT_EQ(39.0, mat(1, 0));
}

TEST_F(DirectSessionMinusAXTest, TestConcurrency) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  // Run the graph 1000 times in 4 different threads concurrently.
  std::vector<string> output_names = {y_ + ":0"};
  auto fn = [&session, output_names]() {
    for (int i = 0; i < 1000; ++i) {
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<Tensor> outputs;
      // Run the graph
      Status s = session->Run(inputs, output_names, {}, &outputs);
      TF_ASSERT_OK(s);
      ASSERT_EQ(1, outputs.size());
      auto mat = outputs[0].matrix<float>();
      EXPECT_FLOAT_EQ(3.0, mat(0, 0));
    }
  };

  for (int i = 0; i < 4; ++i) {
    tp->Schedule(fn);
  }

  // Wait for the functions to finish.
  delete tp;
}

TEST_F(DirectSessionMinusAXTest, TestConcurrency_Callable) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  Session::CallableHandle handle;
  TF_ASSERT_OK(
      session->MakeCallable(MakeCallableOptions({}, {y_ + ":0"}, {}), &handle));

  // Run the callable 1000 times in 4 different threads concurrently.
  auto fn = [&session, handle]() {
    for (int i = 0; i < 1000; ++i) {
      std::vector<Tensor> outputs;
      // Run the graph
      TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
      ASSERT_EQ(1, outputs.size());
      auto mat = outputs[0].matrix<float>();
      EXPECT_FLOAT_EQ(3.0, mat(0, 0));
    }
  };

  for (int i = 0; i < 4; ++i) {
    tp->Schedule(fn);
  }

  // Wait for the functions to finish.
  delete tp;
}

TEST_F(DirectSessionMinusAXTest, TestPerSessionThreads) {
  Initialize({1, 2, 3, 4});

  SessionOptions options;
  options.config.set_use_per_session_threads(true);
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));

  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  // Run the graph 1000 times in 4 different threads concurrently.
  std::vector<string> output_names = {y_ + ":0"};
  auto fn = [&session, output_names]() {
    for (int i = 0; i < 1000; ++i) {
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<Tensor> outputs;
      // Run the graph
      Status s = session->Run(inputs, output_names, {}, &outputs);
      TF_ASSERT_OK(s);
      ASSERT_EQ(1, outputs.size());
      auto mat = outputs[0].matrix<float>();
      EXPECT_FLOAT_EQ(3.0, mat(0, 0));
    }
  };

  for (int i = 0; i < 4; ++i) {
    tp->Schedule(fn);
  }

  // Wait for the functions to finish.
  delete tp;
}

TEST_F(DirectSessionMinusAXTest, TwoCreateCallsFails) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Second is not.
  ASSERT_FALSE(session->Create(def_).ok());
}

TEST_F(DirectSessionMinusAXTest, ForgetToCreate) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;
  ASSERT_FALSE(session->Run(inputs, {y_ + ":0"}, {y_neg_}, &outputs).ok());
}

TEST_F(DirectSessionMinusAXTest, InvalidDevice) {
  GraphDef def;
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  a_tensor.flat<float>().setRandom();
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  x_tensor.flat<float>().setRandom();
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
  // Skip placing y.
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:2");

  test::graph::ToGraphDef(&graph, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  // Should return an error.
  ASSERT_FALSE(session->Create(def).ok());

  // Fix placement and run again
  def.Clear();
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
  test::graph::ToGraphDef(&graph, &def);
  session.reset(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {y->name() + ":0"}, {}, &outputs));
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetworkWithOpts) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  // Prepares RunOptions and RunMetadata
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 0);

  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  TF_ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));

  // Checks RunMetadata is well-formed
  ASSERT_TRUE(run_metadata.has_step_stats());
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 2);
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetworkWithOpts_Callable) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Request two targets: one fetch output and one non-fetched output.
  Session::CallableHandle handle;
  CallableOptions callable_options =
      MakeCallableOptions({}, {y_ + ":0"}, {y_neg_});
  callable_options.mutable_run_options()->set_trace_level(
      RunOptions::FULL_TRACE);
  TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));

  RunMetadata run_metadata;
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 0);

  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, &run_metadata));

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));

  // Checks RunMetadata is well-formed
  ASSERT_TRUE(run_metadata.has_step_stats());
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 2);
}

TEST(DirectSessionTest, KeepsStateAcrossRunsOfSession) {
  GraphDef def;
  Graph g(OpRegistry::Global());
  Node* var = test::graph::Var(&g, DT_FLOAT, TensorShape({10}));
  var->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor twenty(DT_FLOAT, TensorShape({10}));
  for (int i = 0; i < 10; ++i) {
    twenty.flat<float>()(i) = 20.0;
  }

  Node* twenty_node = test::graph::Constant(&g, twenty);
  twenty_node->set_assigned_device_name(
      "/job:localhost/replica:0/task:0/cpu:0");

  Node* init = test::graph::Assign(&g, var, twenty_node);
  init->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;

  // Initialize the variable
  Status s = session->Run(inputs, {init->name()}, {}, &outputs);
  TF_ASSERT_OK(s);

  // Get the variable's data
  s = session->Run(inputs, {var->name() + ":0"}, {}, &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(1, outputs.size());
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_EQ(20.0, outputs[0].flat<float>()(0));
}

TEST(DirectSessionTest, MultipleFeedTest) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  // Fetch without feeding.
  Status s = session->Run(
      {}, {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(1.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(2.0, outputs[1].flat<float>()(0));

  s = session->Run(
      {}, {second_identity->name() + ":0", first_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(2.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(1.0, outputs[1].flat<float>()(0));

  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  Tensor value_22(DT_FLOAT, TensorShape({}));
  value_22.scalar<float>()() = 22.0;

  // Feed [first_const, second_const]
  s = session->Run(
      {{first_const->name(), value_11}, {second_const->name(), value_22}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [second_const, first_const]
  s = session->Run(
      {{second_const->name(), value_22}, {first_const->name(), value_11}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [first_const, first_const]
  s = session->Run(
      {{first_const->name(), value_11}, {first_const->name(), value_22}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  EXPECT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "fed more than once"));
}

TEST(DirectSessionTest, MultipleFeedTest_Callable) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  Session::CallableHandle handle;
  std::vector<Tensor> outputs;

  // Fetch without feeding.
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {}, {first_identity->name() + ":0", second_identity->name() + ":0"},
          {}),
      &handle));
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(1.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(2.0, outputs[1].flat<float>()(0));

  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {}, {second_identity->name() + ":0", first_identity->name() + ":0"},
          {}),
      &handle));
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(2.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(1.0, outputs[1].flat<float>()(0));

  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  Tensor value_22(DT_FLOAT, TensorShape({}));
  value_22.scalar<float>()() = 22.0;

  // Feed [first_const, second_const]
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {first_const->name(), second_const->name()},
          {first_identity->name() + ":0", second_identity->name() + ":0"}, {}),
      &handle));
  TF_ASSERT_OK(
      session->RunCallable(handle, {value_11, value_22}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [second_const, first_const]
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {second_const->name(), first_const->name()},
          {first_identity->name() + ":0", second_identity->name() + ":0"}, {}),
      &handle));
  TF_ASSERT_OK(
      session->RunCallable(handle, {value_22, value_11}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [first_const, first_const]
  Status s = session->MakeCallable(
      MakeCallableOptions(
          {first_const->name(), first_const->name()},
          {first_identity->name() + ":0", second_identity->name() + ":0"}, {}),
      &handle);
  EXPECT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(str_util::StrContains(s.error_message(), "fed more than once"));
}

TEST(DirectSessionTest, TestTensorConnectionUseTwice) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {1.0, 2.0, 3.0, 4.0});
  Node* a = test::graph::Constant(&graph, a_tensor);

  Tensor dummy_tensor(DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&dummy_tensor, {-1.0});

  Node* left = test::graph::Constant(&graph, dummy_tensor);
  Node* right = test::graph::Constant(&graph, dummy_tensor);

  // y = A * x
  Node* y = test::graph::Add(&graph, left, right);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  CallableOptions callable_options;
  // Directly wire the output of node a to the outputs of nodes left
  // and right, making the callable graph into "a + a;".
  TensorConnection* c_left = callable_options.add_tensor_connection();
  c_left->set_from_tensor(a->name() + ":0");
  c_left->set_to_tensor(left->name() + ":0");
  TensorConnection* c_right = callable_options.add_tensor_connection();
  c_right->set_from_tensor(a->name() + ":0");
  c_right->set_to_tensor(right->name() + ":0");

  callable_options.add_fetch(y->name() + ":0");

  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
  ASSERT_EQ(1, outputs.size());
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(2.0, mat(0, 0));
  EXPECT_FLOAT_EQ(4.0, mat(0, 1));
  EXPECT_FLOAT_EQ(6.0, mat(1, 0));
  EXPECT_FLOAT_EQ(8.0, mat(1, 1));
  TF_ASSERT_OK(session->ReleaseCallable(handle));
}

TEST(DirectSessionTest, FetchMultipleTimes) {
  Graph g(OpRegistry::Global());
  Tensor seven_tensor(DT_INT32, TensorShape());
  seven_tensor.flat<int32>()(0) = 7;
  Node* seven_node = test::graph::Constant(&g, seven_tensor);

  GraphDef def;
  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  const std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;

  auto seven = seven_node->name();
  Status s = session->Run(inputs, {seven, seven}, {}, &outputs);
  TF_ASSERT_OK(s);

  EXPECT_EQ(2, outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    const Tensor& t = outputs[i];
    ASSERT_TRUE(t.IsInitialized()) << i;
    EXPECT_EQ(7, t.flat<int32>()(0)) << i;
  }
}

REGISTER_OP("Darth").Input("x: float").Output("y: float").Doc(R"doc(
Darth promises one return value.

x: float
y: float
)doc");

// The DarthOp kernel violates its promise to return one-value.
class DarthOp : public OpKernel {
 public:
  explicit DarthOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {}
};
REGISTER_KERNEL_BUILDER(Name("Darth").Device(DEVICE_CPU), DarthOp);

TEST(DirectSessionTest, DarthKernel) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_FLOAT, TensorShape({}));
  vx.scalar<float>()() = 1.0;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "Darth", x);
  GraphDef def;
  test::graph::ToGraphDef(&g, &def);
  auto sess = CreateSession();
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  auto s = sess->Run({}, {y->name() + ":0"}, {}, &outputs);
  EXPECT_TRUE(errors::IsInternal(s));
}

// Have the Darth op in the graph placed on GPU, but don't run it.
TEST(DirectSessionTest, PlacePrunedGraph) {
  {
    Graph g(OpRegistry::Global());
    Tensor vx(DT_FLOAT, TensorShape({}));
    vx.scalar<float>()() = 1.0;
    Node* x = test::graph::Constant(&g, vx);
    Node* y = test::graph::Unary(&g, "Darth", x);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");
    GraphDef def;
    test::graph::ToGraphDef(&g, &def);

    // By default, we place the entire graph, so we should fail the
    // call to Create.
    SessionOptions options;
    std::unique_ptr<Session> sess(NewSession(options));
    auto s = sess->Create(def);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
  }

  {
    Graph g(OpRegistry::Global());
    Tensor vx(DT_FLOAT, TensorShape({}));
    vx.scalar<float>()() = 1.0;
    Node* x = test::graph::Constant(&g, vx);
    Node* y = test::graph::Unary(&g, "Darth", x);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");
    GraphDef def;
    test::graph::ToGraphDef(&g, &def);

    SessionOptions options;
    // Set the option to place pruned graphs, we should expect this
    // to run.
    options.config.mutable_graph_options()->set_place_pruned_graph(true);
    std::unique_ptr<Session> sess(NewSession(options));
    TF_ASSERT_OK(sess->Create(def));
    std::vector<Tensor> outputs;
    auto s = sess->Run({}, {x->name() + ":0"}, {}, &outputs);
    TF_EXPECT_OK(s);
  }
}

TEST(DirectSessionTest, PartialRunTest) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  Node* third = test::graph::Add(&g, first_identity, second_identity);
  Node* third_identity = test::graph::Identity(&g, third);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  string handle;
  Status s = session->PRunSetup(
      {first_const->name(), second_const->name()},
      {first_identity->name() + ":0", second_identity->name() + ":0",
       third_identity->name() + ":0"},
      {}, &handle);
  TF_ASSERT_OK(s);

  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  Tensor value_22(DT_FLOAT, TensorShape({}));
  value_22.scalar<float>()() = 22.0;

  // Feed first_const, fetch first_identity
  s = session->PRun(handle, {{first_const->name(), value_11}},
                    {first_identity->name() + ":0"}, &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));

  // Feed second_const, fetch second_identity and third_identity
  s = session->PRun(
      handle, {{second_const->name(), value_22}},
      {second_identity->name() + ":0", third_identity->name() + ":0"},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(22.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(11.0 + 22.0, outputs[1].flat<float>()(0));
}

TEST(DirectSessionTest, PartialRunMissingFeed) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  Node* third = test::graph::Add(&g, first_identity, second_identity);
  Node* third_identity = test::graph::Identity(&g, third);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  string handle;
  Status s = session->PRunSetup({first_const->name(), second_const->name()},
                                {third_identity->name() + ":0"}, {}, &handle);
  TF_ASSERT_OK(s);

  // Feed first_const, fetch third_identity
  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  s = session->PRun(handle, {{first_const->name(), value_11}},
                    {third_identity->name() + ":0"}, &outputs);
  ASSERT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                    "can't be computed from the feeds"));
}

TEST(DirectSessionTest, PartialRunMultiOutputFeed) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor bool_value(DT_BOOL, TensorShape({}));
  bool_value.scalar<bool>()() = true;
  Node* bool_const = test::graph::Constant(&g, bool_value);
  Node* switch_node = test::graph::Switch(&g, bool_const, bool_const);
  Node* fourth_identity = test::graph::Identity(&g, switch_node, 1);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  string handle;
  Status s = session->PRunSetup({switch_node->name() + ":1"},
                                {fourth_identity->name() + ":0"}, {}, &handle);
  TF_ASSERT_OK(s);

  // Fetch fourth_identity without feeds.
  s = session->PRun(handle, {}, {fourth_identity->name() + ":0"}, &outputs);
  ASSERT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(str_util::StrContains(s.error_message(),
                                    "can't be computed from the feeds"));

  // Feed switch_node:1 and fetch fourth_identity.
  s = session->PRun(handle, {{switch_node->name() + ":1", bool_value}},
                    {fourth_identity->name() + ":0"}, &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(true, outputs[0].flat<bool>()(0));
}

TEST(DirectSessionTest, RunHandleTest) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor value0(DT_FLOAT, TensorShape({}));
  value0.scalar<float>()() = 1.0;
  Node* const0 = test::graph::Constant(&g, value0);
  Node* identity0 = test::graph::Identity(&g, const0);

  Tensor value1(DT_FLOAT, TensorShape({}));
  value1.scalar<float>()() = 2.0;
  Node* const1 = test::graph::Constant(&g, value1);
  Node* node3 = test::graph::Add(&g, identity0, const1);
  Node* node4 = test::graph::Unary(&g, "GetSessionHandleV2", node3);

  Tensor value2(DT_STRING, TensorShape({}));
  Node* const2 = test::graph::Constant(&g, value2);
  Node* node5 = test::graph::GetSessionTensor(&g, const2);
  Node* node6 = test::graph::Add(&g, node5, const1);

  Node* node7 = test::graph::Unary(&g, "DeleteSessionTensor", const2);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // First run call: Create a handle.
  std::vector<Tensor> outputs;
  Status s = session->Run({}, {node4->name() + ":0"}, {}, &outputs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs.size());

  ResourceHandle resource_handle = outputs[0].scalar<ResourceHandle>()();
  Tensor string_handle(DT_STRING, {});
  string_handle.flat<string>().setConstant(resource_handle.name());

  // Second run call: Use a handle.
  std::vector<Tensor> outputs1;
  s = session->Run({{const2->name(), string_handle}}, {node6->name() + ":0"},
                   {}, &outputs1);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs1.size());
  ASSERT_EQ(5.0, outputs1[0].flat<float>()(0));

  // Third run call: Delete a handle.
  std::vector<Tensor> outputs2;
  s = session->Run({{const2->name(), string_handle}}, {}, {node7->name()},
                   &outputs2);
  ASSERT_TRUE(s.ok());
}

TEST(DirectSessionTest, RunHandleTest_Callable) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor value0(DT_FLOAT, TensorShape({}));
  value0.scalar<float>()() = 1.0;
  Node* const0 = test::graph::Constant(&g, value0);
  Node* identity0 = test::graph::Identity(&g, const0);

  Tensor value1(DT_FLOAT, TensorShape({}));
  value1.scalar<float>()() = 2.0;
  Node* const1 = test::graph::Constant(&g, value1);
  Node* node3 = test::graph::Add(&g, identity0, const1);
  Node* node4 = test::graph::Unary(&g, "GetSessionHandleV2", node3);

  Tensor value2(DT_STRING, TensorShape({}));
  Node* const2 = test::graph::Constant(&g, value2);
  Node* node5 = test::graph::GetSessionTensor(&g, const2);
  Node* node6 = test::graph::Add(&g, node5, const1);

  Node* node7 = test::graph::Unary(&g, "DeleteSessionTensor", const2);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // First run call: Create a handle.
  std::vector<Tensor> outputs;
  Status s = session->Run({}, {node4->name() + ":0"}, {}, &outputs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs.size());

  ResourceHandle resource_handle = outputs[0].scalar<ResourceHandle>()();
  Tensor string_handle(DT_STRING, {});
  string_handle.flat<string>().setConstant(resource_handle.name());

  // Second run call: Use a handle.
  std::vector<Tensor> outputs1;
  s = session->Run({{const2->name(), string_handle}}, {node6->name() + ":0"},
                   {}, &outputs1);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs1.size());
  ASSERT_EQ(5.0, outputs1[0].flat<float>()(0));

  // Third run call: Delete a handle.
  std::vector<Tensor> outputs2;
  s = session->Run({{const2->name(), string_handle}}, {}, {node7->name()},
                   &outputs2);
  ASSERT_TRUE(s.ok());
}

TEST(DirectSessionTest, CreateGraphFailsWhenAssigningAFedVar) {
  Graph graph(OpRegistry::Global());

  Node* a = test::graph::Var(&graph, DT_FLOAT, {});
  Node* b = test::graph::Constant(&graph, {});

  Tensor zero(DT_FLOAT, {});
  test::FillValues<float>(&zero, {0});

  // a = b
  Node* assign = test::graph::Assign(&graph, a, b);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);

  // The graph is invalid since a constant cannot be assigned to a constant.
  // The return Status of session->Run should flag this as an invalid argument.
  std::vector<Tensor> outputs;
  Status s = session->Run({{a->name(), zero}}, {assign->name()}, {}, &outputs);
  ASSERT_TRUE(errors::IsInvalidArgument(s));
}

TEST(DirectSessionTest, TimeoutSession) {
  GraphDef graph;
  // Creates a graph with one FIFOQueue and one dequeue op.
  protobuf::TextFormat::ParseFromString(R"proto(
    node {
      name: 'fifo_queue'
      op: 'FIFOQueue'
      device: '/device:CPU:0'
      attr {
        key: 'capacity'
        value {
          i: 10
        }
      }
      attr {
        key: 'component_types'
        value {
          list {
            type: DT_FLOAT
          }
        }
      }
      attr {
        key: 'container'
        value {
          s: ''
        }
      }
      attr {
        key: 'shapes'
        value {
          list {
          }
        }
      }
      attr {
        key: 'shared_name'
        value {
          s: ''
        }
      }
    }
    node {
      name: 'fifo_queue_Dequeue'
      op: 'QueueDequeue'
      input: 'fifo_queue'
      device: '/device:CPU:0'
      attr {
        key: 'component_types'
        value {
          list {
            type: DT_FLOAT
          }
        }
      }
      attr {
        key: 'timeout_ms'
        value {
          i: -1
        }
      }
    }
    versions {
      producer: 9
    }
  )proto",
                                        &graph);

  {
    // Creates a session with operation_timeout_in_ms set to 100 milliseconds.
    SessionOptions options;
    (*options.config.mutable_device_count())["CPU"] = 2;
    options.config.set_operation_timeout_in_ms(100);

    std::unique_ptr<Session> session(NewSession(options));
    ASSERT_TRUE(session != nullptr);
    TF_ASSERT_OK(session->Create(graph));

    // Verifies that the error code is DEADLINE_EXCEEDED.
    Status s = session->Run({}, {}, {"fifo_queue_Dequeue"}, nullptr);
    ASSERT_EQ(error::DEADLINE_EXCEEDED, s.code());
    TF_ASSERT_OK(session->Close());
  }

  {
    // Creates a session with no operation_timeout_in_ms.
    auto session = CreateSession();
    ASSERT_TRUE(session != nullptr);
    TF_ASSERT_OK(session->Create(graph));
    RunOptions run_options;
    run_options.set_timeout_in_ms(20);
    // Verifies that the error code is DEADLINE_EXCEEDED.
    Status s2 = session->Run(run_options, {}, {}, {"fifo_queue_Dequeue"},
                             nullptr, nullptr);
    ASSERT_EQ(error::DEADLINE_EXCEEDED, s2.code());
    TF_ASSERT_OK(session->Close());
  }
}

// Accesses the cancellation manager for the step after the step has been
// cancelled.
class CancellationMgrPollingOp : public OpKernel {
 public:
  explicit CancellationMgrPollingOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    CancellationManager* cm = ctx->cancellation_manager();
    while (!cm->IsCancelled()) {
      ctx->env()->SleepForMicroseconds(1000);
    }
    notification.Notify();
  }
  static Notification notification;
};
Notification CancellationMgrPollingOp::notification;

REGISTER_KERNEL_BUILDER(Name("CancellationMgrPollingOp").Device(DEVICE_CPU),
                        CancellationMgrPollingOp);
REGISTER_OP("CancellationMgrPollingOp").Doc("");

TEST(DirectSessionTest, TestTimeoutCleanShutdown) {
  GraphDef graph;
  // Creates a graph with one FIFOQueue and one dequeue op.
  protobuf::TextFormat::ParseFromString(R"proto(
    node {
      name: 'cm_polling'
      op: 'CancellationMgrPollingOp'
      device: '/device:CPU:0'
    }
    versions {
      producer: 9
    }
  )proto",
                                        &graph);

  // Creates a session with operation_timeout_in_ms set to 100 milliseconds.
  SessionOptions options;
  options.config.set_operation_timeout_in_ms(100);
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(graph));

  // Verifies that the error code is DEADLINE_EXCEEDED.
  Status s = session->Run({}, {}, {"cm_polling"}, nullptr);
  ASSERT_EQ(error::DEADLINE_EXCEEDED, s.code());

  // Verify that the op ran to completion.
  ASSERT_TRUE(CancellationMgrPollingOp::notification.HasBeenNotified());

  TF_ASSERT_OK(session->Close());
}

static void TestSessionInterOpThreadsImpl(bool use_function_lib,
                                          bool use_global_pools) {
  using test::function::blocking_op_state;
  using test::function::BlockingOpState;

  FunctionDefLibrary library_graph_def;
  if (use_function_lib) {
    *library_graph_def.add_function() = test::function::BlockingOpFn();
  }

  FunctionLibraryDefinition flib(OpRegistry::Global(), library_graph_def);
  Graph g(&flib);
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* x = test::graph::Constant(&g, t);
  Node* y;
  if (use_function_lib) {
    y = test::graph::Unary(&g, "BlockingOpFn", x);
  } else {
    y = test::graph::Unary(&g, "BlockingOp", x);
  }
  GraphDef def;
  test::graph::ToGraphDef(&g, &def);
  *def.mutable_library() = library_graph_def;

  // Create session with two inter-op thread pools.
  SessionOptions options;
  // Turn off optimizations so that the blocking op doesn't get invoked during
  // graph setup.
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  (*options.config.mutable_device_count())["CPU"] = 2;
  (*options.config.mutable_device_count())["GPU"] = 0;
  (*options.config.mutable_device_count())["SYCL"] = 0;

  auto* p = options.config.add_session_inter_op_thread_pool();
  if (use_global_pools) p->set_global_name("large pool");
  p = options.config.add_session_inter_op_thread_pool();
  if (use_global_pools) p->set_global_name("small pool");
  p->set_num_threads(1);
  const int kLargePool = 0;
  const int kSmallPool = 1;

  std::vector<std::unique_ptr<Session>> sessions;
  if (!use_global_pools) {
    sessions.emplace_back(NewSession(options));
    TF_ASSERT_OK(sessions.back()->Create(def));
  }
  mutex sessions_mu;

  std::atomic<int32> num_done(0);
  // Runs session to compute <node>:0 using inter_op thread pool <pool>.
  auto add_session_run_call =
      [use_global_pools, &def, &options, &sessions, &sessions_mu, &num_done](
          thread::ThreadPool* tp, Node* node, int inter_op_pool) {
        auto fn = [use_global_pools, &def, &options, &sessions, &sessions_mu,
                   inter_op_pool, node, &num_done]() {
          RunOptions run_options;
          run_options.set_inter_op_thread_pool(inter_op_pool);
          std::vector<Tensor> outputs;

          Session* session;
          if (use_global_pools) {
            std::unique_ptr<Session> s(NewSession(options));
            TF_ASSERT_OK(s->Create(def));
            session = s.get();

            mutex_lock l(sessions_mu);
            sessions.emplace_back(std::move(s));
          } else {
            session = sessions[0].get();
          }

          Status s = session->Run(run_options, {} /* inputs */,
                                  {node->name() + ":0"} /* output_names */, {},
                                  &outputs, nullptr /* run_metadata */);
          TF_CHECK_OK(s);
          ASSERT_EQ(1, outputs.size());
          auto flat = outputs[0].flat<float>();
          EXPECT_FLOAT_EQ(1.2, flat(0));
          num_done.fetch_add(1);
        };
        tp->Schedule(fn);
      };

  // For blocking states:
  // - Starts at 0, BlockingOp::Compute will move to 1.
  // - This main thread will wait for 1, then move to 2 when other ops are done.
  //   Moving to 2 unblocks the blocking op, which then moves to state 3.

  // Run the graph once on the non-limited pool.
  thread::ThreadPool* tp1 = new thread::ThreadPool(Env::Default(), "tp1", 1);
  blocking_op_state = new BlockingOpState();
  add_session_run_call(tp1, y, kLargePool);
  blocking_op_state->AwaitState(1);
  blocking_op_state->MoveToState(1, 2);
  blocking_op_state->AwaitState(3);
  blocking_op_state->MoveToState(3, 0);
  delete tp1;
  num_done = 0;

  tp1 = new thread::ThreadPool(Env::Default(), "tp1", 5);

  // Launch 2 session run calls. Neither will finish until the blocking op is
  // unblocked, because it is using all threads in the small pool.
  add_session_run_call(tp1, y, kSmallPool);
  blocking_op_state->AwaitState(1);  // Wait for the blocking op to Compute.

  // These will block on <BlockingOpState>.
  const int kBlockedThreads = 3;
  for (int i = 0; i < kBlockedThreads; ++i) {
    add_session_run_call(tp1, x, kSmallPool);
  }

  // Launch session calls using the other inter-op pool. These will finish
  // as they are in inter_op pool #2.
  thread::ThreadPool* tp2 = new thread::ThreadPool(Env::Default(), "tp2", 3);
  const int kUnblockedThreads = 4;
  for (int i = 0; i < kUnblockedThreads; ++i) {
    add_session_run_call(tp2, x, kLargePool);
  }
  delete tp2;
  EXPECT_EQ(kUnblockedThreads, num_done.load());

  // Unblock the blocked op and wait for the blocked functions to finish.
  blocking_op_state->MoveToState(1, 2);
  delete tp1;
  EXPECT_EQ(kUnblockedThreads + kBlockedThreads + 1, num_done.load());
  delete blocking_op_state;
  blocking_op_state = nullptr;
}

TEST(DirectSessionTest, TestSessionInterOpThreads) {
  TestSessionInterOpThreadsImpl(false /* use_function_lib */,
                                false /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpThreadsWithFunctions) {
  TestSessionInterOpThreadsImpl(true /* use_function_lib */,
                                false /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpGlobalPools) {
  TestSessionInterOpThreadsImpl(false /* use_function_lib */,
                                true /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpGlobalPoolsWithFunctions) {
  TestSessionInterOpThreadsImpl(true /* use_function_lib */,
                                true /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpThreadsInvalidOptions) {
  Graph g(OpRegistry::Global());
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* x = test::graph::Constant(&g, t);
  GraphDef def;
  test::graph::ToGraphDef(&g, &def);

  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  (*options.config.mutable_device_count())["CPU"] = 2;

  options.config.add_session_inter_op_thread_pool();

  // Wrong pool number on Run call.
  {
    std::unique_ptr<Session> session(NewSession(options));
    TF_ASSERT_OK(session->Create(def));
    for (int pool_num = -1; pool_num <= 1; pool_num += 2) {
      RunOptions run_options;
      run_options.set_inter_op_thread_pool(pool_num);
      std::vector<Tensor> outputs;
      Status s = session->Run(run_options, {} /* inputs */,
                              {x->name() + ":0"} /* output_names */, {},
                              &outputs, nullptr /* run_metadata */);
      EXPECT_EQ(
          strings::StrCat("Invalid argument: Invalid inter_op_thread_pool: ",
                          pool_num),
          s.ToString());
    }
  }

  // Global name changes thread count.
  std::vector<std::unique_ptr<Session>> sessions;
  auto* pool_config = options.config.mutable_session_inter_op_thread_pool(0);
  pool_config->set_num_threads(0);
  pool_config->set_global_name("foo");
  sessions.emplace_back(NewSession(options));
  TF_ASSERT_OK(sessions.back()->Create(def));
  sessions.emplace_back(NewSession(options));  // repeat creation, okay.
  TF_ASSERT_OK(sessions.back()->Create(def));
  for (int pass = 0; pass < 2; ++pass) {
    for (int i = 1; i < 128; ++i) {
      pool_config->set_num_threads(i);
      sessions.emplace_back(NewSession(options));
      auto status = sessions.back()->Create(def);
      ASSERT_FALSE(status.ok()) << status;
    }

    // Clear existing sessions before second pass; error still happens.
    sessions.clear();
  }
}

TEST(DirectSessionTest, TestDirectSessionRunClose) {
  // Construct a graph with a variable and a single assign.
  Graph g(OpRegistry::Global());
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* var_val = test::graph::Constant(&g, t);
  Node* var = test::graph::Var(&g, DT_FLOAT, {});
  Node* var_assign = test::graph::Assign(&g, var, var_val);
  GraphDef def;
  test::graph::ToGraphDef(&g, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // Assign a value to the var.
  TF_ASSERT_OK(session->Run({} /* inputs */, {},
                            {var_assign->name()} /* target_nodes */, nullptr));

  // Run a read on the variable to ensure that it works.
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(
      {} /* inputs */, {var->name() + ":0"} /* output_names */, {}, &outputs));
  EXPECT_EQ(t.scalar<float>()(), outputs[0].scalar<float>()());
  outputs.clear();

  // Make a callable handle before closing the session.
  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions({}, {}, {var_assign->name()}), &handle));

  // Close the session.
  TF_ASSERT_OK(session->Close());

  // Run the read on the variable to get an error.
  Status s = session->Run({} /* inputs */, {},
                          {var_assign->name()} /* target_nodes */, nullptr);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());

  // Run the read as a callable to verify that we get the same error.
  s = session->RunCallable(handle, {}, {}, nullptr);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());
}

TEST(DirectSessionTest, TestDirectSessionPRunClose) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  Node* third = test::graph::Add(&g, first_identity, second_identity);
  Node* third_identity = test::graph::Identity(&g, third);

  test::graph::ToGraphDef(&g, &def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  string handle;
  Status s = session->PRunSetup(
      {first_const->name(), second_const->name()},
      {first_identity->name() + ":0", second_identity->name() + ":0",
       third_identity->name() + ":0"},
      {}, &handle);
  TF_ASSERT_OK(s);

  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  Tensor value_22(DT_FLOAT, TensorShape({}));
  value_22.scalar<float>()() = 22.0;

  // Close the session.
  TF_ASSERT_OK(session->Close());

  // Feed first_const, fetch first_identity
  s = session->PRun(handle, {{first_const->name(), value_11}},
                    {first_identity->name() + ":0"}, &outputs);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());
}

TEST(DirectSessionTest, TestDirectSessionReset) {
  // Construct a graph with a variable and a single assign.
  Graph g(OpRegistry::Global());
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* var_val = test::graph::Constant(&g, t);
  Node* var = test::graph::Var(&g, DT_FLOAT, {});
  Node* var_assign = test::graph::Assign(&g, var, var_val);
  GraphDef def;
  test::graph::ToGraphDef(&g, &def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // Assign a value to the var.
  TF_ASSERT_OK(session->Run({} /* inputs */, {},
                            {var_assign->name()} /* target_nodes */, nullptr));

  // Run a read on the variable to ensure that it works.
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(
      {} /* inputs */, {var->name() + ":0"} /* output_names */, {}, &outputs));
  EXPECT_EQ(t.scalar<float>()(), outputs[0].scalar<float>()());
  outputs.clear();

  // Reset the containers.
  TF_EXPECT_OK(Reset(options, {}));

  // Run the read on the variable to get an error.
  // TODO(suharshs): This test only works because we close the Session in Reset.
  // If we change the behavior of Reset to not close the Session, this test will
  // fail, since the Variable buffer is cached by var.
  Status s = session->Run({} /* inputs */, {},
                          {var_assign->name()} /* target_nodes */, nullptr);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());
}

TEST(DirectSessionTest, LocalDeviceManager) {
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));

  const DeviceMgr* mgr = nullptr;
  TF_ASSERT_OK(session->LocalDeviceManager(&mgr));
  ASSERT_TRUE(mgr != nullptr);
  EXPECT_GT(mgr->ListDevices().size(), 0);
}

// A simple benchmark for the overhead of `DirectSession::Run()` calls
// with varying numbers of feeds/fetches.
void FeedFetchBenchmarkHelper(int iters, int num_feeds,
                              bool use_make_callable) {
  testing::StopTiming();

  Tensor value(DT_FLOAT, TensorShape());
  value.flat<float>()(0) = 37.0;

  std::vector<std::pair<string, Tensor>> inputs;
  inputs.reserve(num_feeds);
  std::vector<string> outputs;

  Graph g(OpRegistry::Global());
  for (int i = 0; i < num_feeds; ++i) {
    // NOTE(mrry): We pin nodes to the "/cpu:0" device, so as not to
    // measure CPU<->GPU copying overhead. We should also optimize and
    // monitor this overhead where possible, but that is not the
    // object of study in this benchmark.
    Node* placeholder;
    TF_CHECK_OK(NodeBuilder(g.NewName("Placeholder"), "Placeholder")
                    .Attr("shape", TensorShape())
                    .Attr("dtype", DT_FLOAT)
                    .Device("/cpu:0")
                    .Finalize(&g, &placeholder));
    Node* identity;
    TF_CHECK_OK(NodeBuilder(g.NewName("Identity"), "Identity")
                    .Input(placeholder)
                    .Attr("T", DT_FLOAT)
                    .Device("/cpu:0")
                    .Finalize(&g, &identity));
    inputs.push_back({placeholder->name() + ":0", value});
    outputs.push_back(identity->name() + ":0");
  }
  GraphDef gd;
  g.ToGraphDef(&gd);
  SessionOptions opts;
  std::unique_ptr<Session> session(NewSession(opts));
  TF_CHECK_OK(session->Create(gd));
  if (use_make_callable) {
    Session::CallableHandle handle;
    CallableOptions callable_options;
    std::vector<Tensor> input_tensors;
    for (const auto& input : inputs) {
      callable_options.add_feed(input.first);
      input_tensors.push_back(input.second);
    }
    for (const string& output : outputs) {
      callable_options.add_fetch(output);
    }
    TF_CHECK_OK(session->MakeCallable(callable_options, &handle));

    testing::StartTiming();
    for (int i = 0; i < iters; ++i) {
      std::vector<Tensor> output_values;
      TF_CHECK_OK(
          session->RunCallable(handle, input_tensors, &output_values, nullptr));
    }
    testing::StopTiming();
  } else {
    {
      // NOTE(mrry): Ignore the first run, which will incur the graph
      // partitioning/pruning overhead and skew the results.
      //
      // Note that we should also optimize and monitor the overhead on
      // the first run, which will impact application startup times, but
      // that is not the object of study in this benchmark.
      std::vector<Tensor> output_values;
      TF_CHECK_OK(session->Run(inputs, outputs, {}, &output_values));
    }
    testing::StartTiming();
    for (int i = 0; i < iters; ++i) {
      std::vector<Tensor> output_values;
      TF_CHECK_OK(session->Run(inputs, outputs, {}, &output_values));
    }
    testing::StopTiming();
  }
}

void BM_FeedFetch(int iters, int num_feeds) {
  FeedFetchBenchmarkHelper(iters, num_feeds, /* use_make_callable */ false);
}
void BM_FeedFetchCallable(int iters, int num_feeds) {
  FeedFetchBenchmarkHelper(iters, num_feeds, /* use_make_callable */ true);
}

BENCHMARK(BM_FeedFetch)->Arg(1)->Arg(2)->Arg(5)->Arg(10);
BENCHMARK(BM_FeedFetchCallable)->Arg(1)->Arg(2)->Arg(5)->Arg(10);

}  // namespace
}  // namespace tensorflow
