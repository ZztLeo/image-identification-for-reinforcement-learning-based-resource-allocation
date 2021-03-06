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

#include <list>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/java/src/gen/cc/java_defs.h"
#include "tensorflow/java/src/gen/cc/source_writer.h"

namespace tensorflow {
namespace java {
namespace {

TEST(AppendTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2).Append("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.Prefix("--").Append("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2).Prefix("--").Append("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2).Write("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.Prefix("--").Write("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2).Prefix("--").Write("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\n  --and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Basic) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye").EndLine().Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Indent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Append("and I say hello!");

  const char* expected = "You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, IndentAndOutdent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Append("and I say hello!")
      .EndLine()
      .Indent(-2)
      .Append("Hello, hello!");

  const char* expected = "You say goodbye\n  and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Prefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Prefix("--")
      .Append("and I say hello!");

  const char* expected = "You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, PrefixAndRemovePrefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Prefix("--")
      .Append("and I say hello!")
      .EndLine()
      .Prefix("")
      .Append("Hello, hello!");

  const char* expected = "You say goodbye\n--and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, IndentAndPrefixAndOutdentAndRemovePrefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Prefix("--")
      .Append("and I say hello!")
      .EndLine()
      .Indent(-2)
      .Prefix("")
      .Append("Hello, hello!");

  const char* expected = "You say goodbye\n  --and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, NegativeIndent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(-10)
      .Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, CumulativeIndent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Append("and I say hello!")
      .EndLine()
      .Indent(2)
      .Append("Hello, hello!");

  const char* expected =
      "You say goodbye\n  and I say hello!\n    Hello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, EmptyPrefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Prefix("")
      .Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(StreamTest, BlocksAndLines) {
  SourceBufferWriter writer;

  writer.Append("int i = 0;").EndLine()
        .Append("int j = 10;").EndLine()
        .Append("if (true)")
        .BeginBlock()
          .Append("int aLongWayToTen = 0;").EndLine()
          .Append("while (++i <= j)")
          .BeginBlock()
            .Append("++aLongWayToTen;").EndLine()
          .EndBlock()
        .EndBlock();

  const char* expected =
      "int i = 0;\n"
      "int j = 10;\n"
      "if (true) {\n"
      "  int aLongWayToTen = 0;\n"
      "  while (++i <= j) {\n"
      "    ++aLongWayToTen;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(StreamTest, Types) {
  SourceBufferWriter writer;
  Type generic = Type::Generic("T").add_supertype(Type::Class("Number"));

  writer.AppendType(Type::Int()).Append(", ")
        .AppendType(Type::Class("String")).Append(", ")
        .AppendType(generic).Append(", ")
        .AppendType(Type::ListOf(generic)).Append(", ")
        .AppendType(Type::ListOf(Type::IterableOf(generic))).Append(", ")
        .AppendType(Type::ListOf(Type::Generic()));

  const char* expected =
      "int, String, T, List<T>, List<Iterable<T>>, List<?>";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(StreamTest, FileSnippet) {
  SourceBufferWriter writer;
  const string fname = tensorflow::io::JoinPath(
      tensorflow::testing::TensorFlowSrcRoot(),
      "java/src/gen/resources/test.java.snippet");

  writer.WriteFromFile(fname)
        .BeginBlock()
        .WriteFromFile(fname)
        .EndBlock();

  const char* expected =
      "// Here is a little snippet\n"
      "System.out.println(\"Hello!\");\n"
      "{\n"
      "  // Here is a little snippet\n"
      "  System.out.println(\"Hello!\");\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, SimpleClass) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");

  writer.BeginType(clazz, nullptr, PUBLIC).EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, SimpleClassWithDependencies) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  std::list<Type> deps;
  deps.push_back(Type::Class("TypeA", "org.test.sub"));
  deps.push_back(Type::Class("TypeA", "org.test.sub"));  // a second time
  deps.push_back(Type::Class("TypeB", "org.other"));
  deps.push_back(Type::Class("SamePackageType", "org.tensorflow"));
  deps.push_back(Type::Class("NoPackageType"));

  writer.BeginType(clazz, &deps, PUBLIC).EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "import org.other.TypeB;\n"
      "import org.test.sub.TypeA;\n\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, AnnotatedAndDocumentedClass) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  clazz.description("This class has a\n<p>\nmultiline description.");
  clazz.add_annotation(Annotation::Create("Bean"));
  clazz.add_annotation(Annotation::Create("SuppressWarnings")
      .attributes("\"rawtypes\""));

  writer.BeginType(clazz, nullptr, PUBLIC).EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "/**\n"
      " * This class has a\n"
      " * <p>\n"
      " * multiline description.\n"
      " **/\n"
      "@Bean\n"
      "@SuppressWarnings(\"rawtypes\")\n"
      "public class Test {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, ParameterizedClass) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  clazz.add_parameter(Type::Generic("T"));
  clazz.add_parameter(Type::Generic("U").add_supertype(Type::Class("Number")));

  writer.BeginType(clazz, nullptr, PUBLIC).EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test<T, U extends Number> {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, ParameterizedClassAndSupertypes) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Type type_t = Type::Generic("T");
  clazz.add_parameter(type_t);
  Type type_u = Type::Generic("U").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_u);
  clazz.add_supertype(Type::Interface("Parametrizable").add_parameter(type_u));
  clazz.add_supertype(Type::Interface("Runnable"));
  clazz.add_supertype(Type::Class("SuperTest").add_parameter(type_t));

  writer.BeginType(clazz, nullptr, PUBLIC).EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test<T, U extends Number>"
      " extends SuperTest<T> implements Parametrizable<U>, Runnable {\n}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, ParameterizedClassFields) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  std::list<Variable> static_fields;
  static_fields.push_back(Variable::Create("field1", Type::Class("String")));
  std::list<Variable> member_fields;
  member_fields.push_back(Variable::Create("field2", Type::Class("String")));
  member_fields.push_back(Variable::Create("field3", type_t));

  writer.BeginType(clazz, nullptr, PUBLIC)
          .WriteFields(static_fields, STATIC | PUBLIC | FINAL)
          .WriteFields(member_fields, PRIVATE)
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static final String field1;\n"
      "  \n"
      "  private String field2;\n"
      "  private T field3;\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, SimpleInnerClass) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Type inner_class = Type::Class("InnerTest");

  writer.BeginType(clazz, nullptr, PUBLIC)
          .BeginInnerType(inner_class, PUBLIC)
          .EndType()
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test {\n"
      "  \n"
      "  public class InnerTest {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteType, StaticParameterizedInnerClass) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  Type inner_class = Type::Class("InnerTest");
  inner_class.add_parameter(type_t);

  writer.BeginType(clazz, nullptr, PUBLIC)
          .BeginInnerType(inner_class, PUBLIC | STATIC)
          .EndType()
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static class InnerTest<T extends Number> {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, SimpleMethod) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Method method = Method::Create("doNothing", Type::Void());

  writer.BeginType(clazz, nullptr, PUBLIC)
          .BeginMethod(method, PUBLIC).EndMethod()
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test {\n"
      "  \n"
      "  public void doNothing() {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, AnnotatedAndDocumentedMethod) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Method method = Method::Create("doNothing", Type::Void());
  method.description("This method has a\n<p>\nmultiline description.");
  method.add_annotation(Annotation::Create("Override"));
  method.add_annotation(Annotation::Create("SuppressWarnings")
      .attributes("\"rawtypes\""));

  writer.BeginType(clazz, nullptr, PUBLIC)
          .BeginMethod(method, PUBLIC).EndMethod()
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test {\n"
      "  \n"
      "  /**\n"
      "   * This method has a\n"
      "   * <p>\n"
      "   * multiline description.\n"
      "   **/\n"
      "  @Override\n"
      "  @SuppressWarnings(\"rawtypes\")\n"
      "  public void doNothing() {\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, DocumentedMethodWithArguments) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Method method = Method::Create("boolToInt", Type::Int());
  method.description("Converts a boolean to an int");
  method.return_description("int value for this boolean");
  method.add_argument(Variable::Create("b", Type::Boolean()));
  Variable reverse = Variable::Create("reverse", Type::Boolean());
  reverse.description("if true, value is reversed");
  method.add_argument(reverse);

  writer.BeginType(clazz, nullptr, PUBLIC)
          .BeginMethod(method, PUBLIC)
            .Append("if (b && !reverse)")
            .BeginBlock()
              .Append("return 1;").EndLine()
            .EndBlock()
          .Append("return 0;").EndLine()
          .EndMethod()
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test {\n"
      "  \n"
      "  /**\n"
      "   * Converts a boolean to an int\n"
      "   * \n"
      "   * @param b\n"
      "   * @param reverse if true, value is reversed\n"
      "   * @return int value for this boolean\n"
      "   **/\n"
      "  public int boolToInt(boolean b, boolean reverse) {\n"
      "    if (b && !reverse) {\n"
      "      return 1;\n"
      "    }\n"
      "    return 0;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, ParameterizedMethod) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  Method method = Method::Create("doNothing", type_t);

  writer.BeginType(clazz, nullptr, PUBLIC)
          .BeginMethod(method, PUBLIC)
            .Append("return null;").EndLine()
          .EndMethod()
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public T doNothing() {\n"
      "    return null;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteMethod, StaticParameterizedMethod) {
  SourceBufferWriter writer;
  Type clazz = Type::Class("Test", "org.tensorflow");
  Type type_t = Type::Generic("T").add_supertype(Type::Class("Number"));
  clazz.add_parameter(type_t);
  Method method = Method::Create("doNothing", type_t);

  writer.BeginType(clazz, nullptr, PUBLIC)
          .BeginMethod(method, PUBLIC | STATIC)
            .Append("return null;").EndLine()
          .EndMethod()
        .EndType();

  const char* expected =
      "package org.tensorflow;\n\n"
      "public class Test<T extends Number> {\n"
      "  \n"
      "  public static <T extends Number> T doNothing() {\n"
      "    return null;\n"
      "  }\n"
      "}\n";
  ASSERT_STREQ(expected, writer.str().data());
}

}  // namespace
}  // namespace java
}  // namespace tensorflow
