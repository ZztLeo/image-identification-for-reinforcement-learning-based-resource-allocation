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

// SWIG typemaps and declarations for building, compiling, and
// executing XLA computations, wrapping most of what is declared in
// local_computation_builder.h.
//
// The typemaps below implement/assert the following correspondences
// (with elaborations below):
//
//    C++                                  Python
// -------------------------------------+---------------------------------------
//  ComputationDataHandle              <-> int
//  ArraySlice<int64>                  <-  sequence of int
//  ArraySlice<ComputationDataHandle>  <-  sequence of int
//  Literal                            <-> (nested tuple of) numpy ndarray
//  std::vector<Literal>               <-  sequence of (nested tuple of) ndarray
//  Shape                               -> pair holding (dtype, dimensions)
//                                     <-  object duck-typed as xla_client.Shape
//  std::vector<Shape>                 <-  sequence of xla_client.Shape objects
//  PrimitiveType                      <-  int
//  ArraySlice<pair<int64, in64>>      <-  sequence of int pairs
//  PaddingConfig proto                <-  corresponding Python proto
//  ConvolutionDimensionNumbers proto  <-  corresponding Python proto
//  DotDimensionNumbers proto          <-  corresponding Python proto
//
// Arrows indicate whether a conversion only ever occurs in one
// direction, or whether it is maintained bidirectionally.
//
// The Python objects corresponding to C++ Literals have the type:
//
//   T = ndarray | (T, ...)
//
// where a terminal numpy ndarray translates to a Literal with a
// non-tuple Shape, an XLA primitive element type corresponding to the
// ndarray's dtype. Meanwhile, a non-terminal "tuple of T" translates
// to a tuple-shaped Literal whose tuple components are translated
// recursively. For example, if x is a numpy ndarray in Python, with
// shape (2, 3) and dtype of dtype('float32'), then x translates to a
// Literal with rank 2, dimension 2 and 3, and XLA primitive type
// F32. Meanwhile,
//
//   (x, (x, x), (x,)),
//
// translates to a tuple-shaped XLA Literal, whose component subshapes
// are a 2x3 F32-shaped literal followed by two tuple-shaped literals.
//
// Shapes output by C++ become Python objects with the type:
//
//   T            = (dtype, S)
//   S            = DIMENSIONS | TUPLE_SHAPES
//   DIMENSIONS   = (int, ...)
//   TUPLE_SHAPES = (T, ...)
//
// In the pair described by the T rule, the terminal dtype determines
// whether S expands as DIMENSIONS or TUPLE_SHAPES. Namely if it is
// dtype('O'), numpy's object dtype, the structure represents a tuple
// shape and the expansion of the non-terminal S is
// TUPLE_SHAPES. Otherwise, dtype describes a primitive element type
// and S expands into DIMENSIONS giving dimension sizes. For example:
//
//   (dtype('float32'), (3, 5, 7))
//
// describes a 3x5x7 array of F32s, and
//
//   (dtype('O'), ((dtype('float32'), (2, 3)),
//                 (dtype('float64'), (4, 5))))
//
// describes a tuple shape with two subshapes: the first a 2x3 F32,
// and the other a 4x5 F64.
//
// The Python int corresponding to a PrimitiveType enum must be valid
// per xla_data.proto (e.g. xla_data.PRED, xla_data.F32).
//
// The SWIG object wrappers generated by this file are not intended
// for end use, but rather for internal use in the Python XLA client,
// xla_client.py.
//
// One central reason for the Python-side indirection is that the
// Python-side objects produced by the typemaps in this file are
// further packaged up by xla_client before being passed on. For
// instance, xla_client wraps the long produced for a C++
// ComputationDataHandle in a Python ComputationDataHandle proto,
// rather than exposing a raw long outside of the client. Similarly,
// the Python pair produced for a C++ Shape is further wrapped in a
// Python class (xla_client.Shape) so as not to expose the raw pair
// externally.
//
// Other SWIG object wrappers (e.g. of LocalComputation) are further
// wrapped by xla_client in order to set up a custom destructor that
// triggers memory deallocation on the C++ side.

%module(threads="1") local_computation_builder

// Keep the GIL except where explicitly specified.
%nothread;

%include "tensorflow/python/platform/base.i"

%{
// Must be included first
#include "tensorflow/python/lib/core/numpy.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/compiler/xla/python/numpy_bridge.h"
#include "tensorflow/compiler/xla/python/local_computation_builder.h"

using namespace xla;
using namespace xla::swig;

namespace xla {
namespace swig {

bool GetIntAttr(PyObject* o, const char* field, int64* result) {
  PyObject* fo = PyObject_GetAttrString(o, field);
  if (!fo) {
    return false;
  }
  const int64 value = numpy::PyIntOrPyLongToLong(fo);
  if (value == -1 && PyErr_Occurred()) {
    Py_DECREF(fo);
    return false;
  }
  Py_DECREF(fo);
  *result = value;
  return true;
}

// Returns "ok"; true if there is no error, false if there was an error.
bool HandleStringAttribute(PyObject* o,
                           const char* attr_name,
                           std::function<void(string s)> f) {
  if (!PyObject_HasAttrString(o, attr_name)) {
    return true;  // It's ok for the object to not have the attribute.
  }
  PyObject* attr = PyObject_GetAttrString(o, attr_name);
  if (attr == nullptr) {
    return false;  // An error occurred getting the attribute.
  }
  if (attr == Py_None) {
    Py_DECREF(attr);
    return true;  // The attribute is None, which we consider ok.
  }
  if (!PyString_Check(attr)) {
    string message = tensorflow::strings::Printf("%s must be a string or none; got %s",
        attr_name, numpy::PyObjectCppRepr(attr).c_str());
    PyErr_SetString(PyExc_TypeError, message.c_str());
    Py_DECREF(attr);
    return false;  // Type error, not ok.
  }
  f(PyString_AsString(attr));
  Py_DECREF(attr);
  return true;  // Handled string attribute, ok!
}

}
}
%}

// Required to use PyArray_* functions.
%init %{
tensorflow::ImportNumpy();
%}

// ComputationDataHandle

%typemap(in) const ComputationDataHandle& (ComputationDataHandle temp) {
  const int64 handle = numpy::PyIntOrPyLongToLong($input);
  if (handle == -1 && PyErr_Occurred()) {
    SWIG_fail;
  }
  temp.set_handle(handle);
  $1 = &temp;
}

%typemap(out) ComputationDataHandle {
  $result = numpy::LongToPyIntOrPyLong($1.handle());
}

%typemap(out) StatusOr<xla::swig::CompiledLocalComputation*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::CompiledLocalComputation*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr< std::unique_ptr<Literal> > {
  if ($1.ok()) {
    std::unique_ptr<Literal> value = $1.ConsumeValueOrDie();
    $result = numpy::PyObjectFromXlaLiteral(*value);
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<xla::swig::LocalComputation*> {
  if ($1.ok()) {
    auto* value = $1.ValueOrDie();
    {
      auto* $1 = value;
      $typemap(out, xla::swig::LocalComputation*)
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<Shape> {
  if ($1.ok()) {
    $result = numpy::PyShapeInfoFromXlaShape($1.ConsumeValueOrDie());
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) StatusOr<bool> {
  if ($1.ok()) {
    $result = PyBool_FromLong($1.ConsumeValueOrDie());
  } else {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
}

%typemap(out) Status {
  if (!$1.ok()) {
    PyErr_SetString(
        PyExc_RuntimeError, $1.ToString().c_str());
    SWIG_fail;
  }
  Py_INCREF(Py_None);
  $result = Py_None;
}

// ArraySlice<int64>

%typemap(in) tensorflow::gtl::ArraySlice<int64>
    (std::vector<int64> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.resize(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    PyObject* py_int = numpy::PyNumberToPyInt(o);
    if (!py_int) {
      PyErr_SetString(
          PyExc_TypeError,
          "Argument sequence element cannot be converted to int");
      Py_DECREF(o);
      SWIG_fail;
    }
    temps[i] = numpy::PyIntOrPyLongToLong(py_int);
    if (temps[i] == -1 && PyErr_Occurred()) {
      Py_DECREF(py_int);
      Py_DECREF(o);
      SWIG_fail;
    }
    Py_DECREF(py_int);
    Py_DECREF(o);
  }
  $1 = temps;
}

// ComputationDataHandle

%typemap(in) tensorflow::gtl::ArraySlice<ComputationDataHandle>
    (std::vector<ComputationDataHandle> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.resize(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    PyObject* py_int = numpy::PyNumberToPyInt(o);
    if (!py_int) {
      PyErr_SetString(
          PyExc_TypeError,
          "Argument sequence element cannot be converted to int");
      SWIG_fail;
    }
    const int64 handle = numpy::PyIntOrPyLongToLong(py_int);
    if (handle == -1 && PyErr_Occurred()) {
      Py_DECREF(py_int);
      Py_DECREF(o);
      SWIG_fail;
    }
    temps[i].set_handle(handle);
    Py_DECREF(py_int);
    Py_DECREF(o);
  }
  $1 = temps;
}

// LocalShapedBuffer*

%typemap(in) tensorflow::gtl::ArraySlice<xla::swig::LocalShapedBuffer*>
    (std::vector<LocalShapedBuffer*> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    LocalShapedBuffer* lsbp;
    if ((SWIG_ConvertPtr(o, (void**) &lsbp, $descriptor(xla::swig::LocalShapedBuffer*),
                         SWIG_POINTER_EXCEPTION)) == -1) {
      SWIG_fail;
    }
    temps.push_back(lsbp);
    Py_DECREF(o);
  }
  $1 = temps;
}

// Literal

%typemap(in) const Literal& (StatusOr< std::unique_ptr<Literal> > literal_status) {
  literal_status = numpy::XlaLiteralFromPyObject($input);
  if (!literal_status.ok()) {
    PyErr_SetString(PyExc_RuntimeError, literal_status.status().ToString().c_str());
    SWIG_fail;
  }
  $1 = literal_status.ValueOrDie().get();
}

%typemap(out) std::unique_ptr<Literal> {
  $result = numpy::PyObjectFromXlaLiteral(*$1);
}

%typemap(out) StatusOr< std::unique_ptr<Literal> > {
  if (!$1.ok()) {
    PyErr_SetString(PyExc_RuntimeError, $1.status().ToString().c_str());
    SWIG_fail;
  }
  $result = numpy::PyObjectFromXlaLiteral(*$1.ValueOrDie());
}

%typemap(in) const std::vector<Literal>& (std::vector<Literal> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    StatusOr< std::unique_ptr<Literal> > literal_status = numpy::XlaLiteralFromPyObject(o);
    if (!literal_status.ok()) {
      PyErr_SetString(PyExc_RuntimeError, literal_status.status().ToString().c_str());
      Py_DECREF(o);
      SWIG_fail;
    }
    temps.push_back(std::move(*literal_status.ConsumeValueOrDie()));
    Py_DECREF(o);
  }
  $1 = &temps;
}

// OpMetadata

%typemap(in) const OpMetadata& (OpMetadata temp) {
  StatusOr<OpMetadata> statusor = numpy::OpMetadataFromPyObject($input);
  if (!statusor.ok()) {
    PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
    SWIG_fail;
  }
  temp = std::move(statusor).ValueOrDie();
  $1 = &temp;
}

// Shape

%typemap(in) const Shape& (Shape temp) {
  StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape($input);
  if (!statusor.ok()) {
    PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
    SWIG_fail;
  }
  temp = std::move(statusor).ValueOrDie();
  $1 = &temp;
}

%typemap(in) const tensorflow::gtl::optional<Shape>& (
    tensorflow::gtl::optional<Shape> temp) {
  if ($input == Py_None) {
    temp = tensorflow::gtl::nullopt;
    $1 = &temp;
  } else {
    StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape($input);
    if (!statusor.ok()) {
      PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
      SWIG_fail;
    }
    temp = std::move(statusor).ValueOrDie();
    $1 = &temp;
  }
}

%typemap(out) std::unique_ptr<Shape> {
  $result = numpy::PyShapeInfoFromXlaShape(*$1);
}

%typemap(in) const std::vector<Shape>& (std::vector<Shape> temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape(o);
    Py_DECREF(o);
    if (!statusor.ok()) {
      PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
      SWIG_fail;
    }
    temps.push_back(statusor.ConsumeValueOrDie());
  }
  $1 = &temps;
}

%typemap(in) const std::vector<tensorflow::gtl::optional<Shape> >& (
    std::vector<tensorflow::gtl::optional<Shape> > temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    if (o == Py_None) {
      temps.push_back(tensorflow::gtl::nullopt);
    } else {
      StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape(o);
      Py_DECREF(o);
      if (!statusor.ok()) {
        PyErr_SetString(PyExc_RuntimeError, statusor.status().ToString().c_str());
        SWIG_fail;
      }
      temps.push_back(statusor.ConsumeValueOrDie());
    }
  }
  $1 = &temps;
}

// PrimitiveType

%typemap(in) PrimitiveType {
  PyObject* py_int = numpy::PyNumberToPyInt($input);
  if (!py_int) {
    PyErr_SetString(PyExc_TypeError, "Argument cannot be converted to int");
    SWIG_fail;
  }
  const long value = numpy::PyIntOrPyLongToLong(py_int);
  if (value == -1 && PyErr_Occurred()) {
    Py_DECREF(py_int);
    SWIG_fail;
  }
  if (!PrimitiveType_IsValid(value)) {
    PyErr_SetString(
        PyExc_TypeError, "Argument not valid for PrimitiveType enum");
    Py_DECREF(py_int);
    SWIG_fail;
  }
  $1 = static_cast<PrimitiveType>(value);
}

// ArraySlice<pair<int64, in64>>

%typemap(in) tensorflow::gtl::ArraySlice<std::pair<int64, int64> >
    (std::vector<std::pair<int64, int64> > temps) {
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Argument is not a sequence");
    SWIG_fail;
  }
  const int size = PySequence_Size($input);
  temps.reserve(size);
  for (int i = 0; i < size; ++i) {
    PyObject* o = PySequence_GetItem($input, i);
    if (!o) {
      SWIG_fail;
    }
    PyObject* first = PyTuple_GetItem(o, 0);
    if (!first) {
      Py_DECREF(o);
      SWIG_fail;
    }
    PyObject* first_pyint = numpy::PyNumberToPyInt(first);
    if (!first_pyint) {
      PyErr_SetString(
          PyExc_TypeError,
          "First pair item cannot be converted to int");
      Py_DECREF(o);
      SWIG_fail;
    }
    PyObject* second = PyTuple_GetItem(o, 1);
    if (!second) {
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      SWIG_fail;
    }
    PyObject* second_pyint = numpy::PyNumberToPyInt(second);
    if (!second_pyint) {
      PyErr_SetString(
          PyExc_TypeError,
          "Second pair item cannot be converted to int");
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      SWIG_fail;
    }
    const int64 first_value = numpy::PyIntOrPyLongToLong(first_pyint);
    if (first_value == -1 && PyErr_Occurred()) {
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      Py_DECREF(second_pyint);
      SWIG_fail;
    }
    const int64 second_value = numpy::PyIntOrPyLongToLong(second_pyint);
    if (second_value == -1 && PyErr_Occurred()) {
      Py_DECREF(o);
      Py_DECREF(first_pyint);
      Py_DECREF(second_pyint);
      SWIG_fail;
    }
    temps.push_back(std::make_pair(first_value, second_value));
    Py_DECREF(o);
  }
  $1 = temps;
}

// DotDimensionNumbers

%typemap(in) const DotDimensionNumbers&
    (DotDimensionNumbers dimension_numbers) {
  int length;

  /* lhs_contracting_dimensions */
  PyObject* lhs_contracting_dimensions = PyObject_GetAttrString(
      $input, "lhs_contracting_dimensions");
  if (!lhs_contracting_dimensions) {
    SWIG_fail;
  }

  length = PySequence_Size(lhs_contracting_dimensions);
  if (length == -1) {
    Py_DECREF(lhs_contracting_dimensions);
    SWIG_fail;
  }

  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(lhs_contracting_dimensions, i);
    if (!item) {
      Py_DECREF(lhs_contracting_dimensions);
      SWIG_fail;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(lhs_contracting_dimensions);
      SWIG_fail;
    }
    dimension_numbers.add_lhs_contracting_dimensions(dimension);
    Py_DECREF(item);
  }
  Py_DECREF(lhs_contracting_dimensions);

  /* rhs_contracting_dimensions */
  PyObject* rhs_contracting_dimensions = PyObject_GetAttrString(
      $input, "rhs_contracting_dimensions");
  if (!lhs_contracting_dimensions) {
    SWIG_fail;
  }

  length = PySequence_Size(rhs_contracting_dimensions);
  if (length == -1) {
    Py_DECREF(rhs_contracting_dimensions);
    SWIG_fail;
  }

  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(rhs_contracting_dimensions, i);
    if (!item) {
      Py_DECREF(rhs_contracting_dimensions);
      SWIG_fail;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(rhs_contracting_dimensions);
      SWIG_fail;
    }
    dimension_numbers.add_rhs_contracting_dimensions(dimension);
    Py_DECREF(item);
  }
  Py_DECREF(rhs_contracting_dimensions);

  /* lhs_batch_dimensions */
  PyObject* lhs_batch_dimensions = PyObject_GetAttrString(
      $input, "lhs_batch_dimensions");
  if (!lhs_batch_dimensions) {
    SWIG_fail;
  }

  length = PySequence_Size(lhs_batch_dimensions);
  if (length == -1) {
    Py_DECREF(lhs_batch_dimensions);
    SWIG_fail;
  }

  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(lhs_batch_dimensions, i);
    if (!item) {
      Py_DECREF(lhs_batch_dimensions);
      SWIG_fail;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(lhs_batch_dimensions);
      SWIG_fail;
    }
    dimension_numbers.add_lhs_batch_dimensions(dimension);
    Py_DECREF(item);
  }
  Py_DECREF(lhs_batch_dimensions);

  /* rhs_batch_dimensions */
  PyObject* rhs_batch_dimensions = PyObject_GetAttrString(
      $input, "rhs_batch_dimensions");
  if (!rhs_batch_dimensions) {
    SWIG_fail;
  }

  length = PySequence_Size(rhs_batch_dimensions);
  if (length == -1) {
    Py_DECREF(rhs_batch_dimensions);
    SWIG_fail;
  }

  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(rhs_batch_dimensions, i);
    if (!item) {
      Py_DECREF(rhs_batch_dimensions);
      SWIG_fail;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(rhs_batch_dimensions);
      SWIG_fail;
    }
    dimension_numbers.add_rhs_batch_dimensions(dimension);
    Py_DECREF(item);
  }
  Py_DECREF(rhs_batch_dimensions);

  $1 = &dimension_numbers;
}

// PaddingConfig

%typemap(in) const PaddingConfig&
    (PaddingConfig padding_config) {
  PyObject* dimensions = PyObject_GetAttrString($input, "dimensions");
  if (!dimensions) {
    SWIG_fail;
  }

  int length = PySequence_Size(dimensions);
  if (length == -1) {
    Py_DECREF(dimensions);
    SWIG_fail;
  }

  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(dimensions, i);
    if (!item) {
      Py_DECREF(dimensions);
      SWIG_fail;
    }
    int64 edge_padding_low, edge_padding_high, interior_padding;
    if (!GetIntAttr(item, "edge_padding_low", &edge_padding_low)
        || !GetIntAttr(item, "edge_padding_high", &edge_padding_high)
        || !GetIntAttr(item, "interior_padding", &interior_padding)) {
      Py_DECREF(item);
      Py_DECREF(dimensions);
      SWIG_fail;
    }
    Py_DECREF(item);

    PaddingConfig::PaddingConfigDimension* dimension =
        padding_config.add_dimensions();
    dimension->set_edge_padding_low(edge_padding_low);
    dimension->set_edge_padding_high(edge_padding_high);
    dimension->set_interior_padding(interior_padding);
  }
  Py_DECREF(dimensions);

  $1 = &padding_config;
}

// ConvolutionDimensionNumbers

%typemap(in) const ConvolutionDimensionNumbers&
    (ConvolutionDimensionNumbers dimension_numbers) {
  int64 value;

  if (!GetIntAttr($input, "input_batch_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_input_batch_dimension(value);

  if (!GetIntAttr($input, "input_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_input_feature_dimension(value);

  if (!GetIntAttr($input, "output_batch_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_output_batch_dimension(value);

  if (!GetIntAttr($input, "output_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_output_feature_dimension(value);

  if (!GetIntAttr($input, "kernel_output_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_kernel_output_feature_dimension(value);

  if (!GetIntAttr($input, "kernel_input_feature_dimension", &value)) {
    SWIG_fail;
  }
  dimension_numbers.set_kernel_input_feature_dimension(value);

  PyObject* o;
  int length;

  o = PyObject_GetAttrString($input, "input_spatial_dimensions");
  if (!o) {
    SWIG_fail;
  }
  length = PySequence_Size(o);
  if (length == -1) {
    Py_DECREF(o);
    SWIG_fail;
  }
  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(o, i);
    if (!item) {
      Py_DECREF(o);
      SWIG_fail;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(o);
      SWIG_fail;
    }
    dimension_numbers.add_input_spatial_dimensions(dimension);
    Py_DECREF(item);
  }
  Py_DECREF(o);

  o = PyObject_GetAttrString($input, "kernel_spatial_dimensions");
  if (!o) {
    SWIG_fail;
  }
  length = PySequence_Size(o);
  if (length == -1) {
    Py_DECREF(o);
    SWIG_fail;
  }
  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(o, i);
    if (!item) {
      Py_DECREF(o);
      SWIG_fail;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(o);
      SWIG_fail;
    }
    dimension_numbers.add_kernel_spatial_dimensions(dimension);
    Py_DECREF(item);
  }
  Py_DECREF(o);

  o = PyObject_GetAttrString($input, "output_spatial_dimensions");
  if (!o) {
    SWIG_fail;
  }
  length = PySequence_Size(o);
  if (length == -1) {
    Py_DECREF(o);
    SWIG_fail;
  }
  for (int i = 0; i < length; ++i) {
    PyObject* item = PySequence_GetItem(o, i);
    if (!item) {
      Py_DECREF(o);
      SWIG_fail;
    }
    const int64 dimension = numpy::PyIntOrPyLongToLong(item);
    if (dimension == -1 && PyErr_Occurred()) {
      Py_DECREF(item);
      Py_DECREF(o);
      SWIG_fail;
    }
    dimension_numbers.add_output_spatial_dimensions(dimension);
    Py_DECREF(item);
  }
  Py_DECREF(o);

  $1 = &dimension_numbers;
}

// ExecutableBuildOptions

%typemap(in) const ExecutableBuildOptions*
    (ExecutableBuildOptions build_options) {
  if ($input == Py_None) {
    $1 = NULL;
  } else {
    if (!HandleStringAttribute($input, "generate_hlo_graph", [&](string s) {
      build_options.set_generate_hlo_graph(std::move(s));
    })) {
      return nullptr;
    }
    if (!HandleStringAttribute($input, "dump_optimized_hlo_proto_to", [&](string s) {
      build_options.set_dump_optimized_hlo_proto_to(std::move(s));
    })) {
      return nullptr;
    }
    if (!HandleStringAttribute($input, "dump_per_pass_hlo_proto_to", [&](string s) {
      build_options.set_dump_per_pass_hlo_proto_to(std::move(s));
    })) {
      return nullptr;
    }

    PyObject* o = PyObject_GetAttrString($input, "hlo_profile");
    if (o == NULL) {
      SWIG_fail;
    }
    if (o != Py_None) {
      if (!PyBool_Check(o)) {
        PyErr_SetString(PyExc_TypeError, "ExecutableBuildOptions.hlo_profile must be a bool or None.");
        SWIG_fail;
      }
      build_options.set_hlo_profile(o == Py_True);
    }
    Py_DECREF(o);

    o = PyObject_GetAttrString($input, "result_shape");
    if (o == nullptr) {
      return nullptr;
    }
    if (o != Py_None) {
      StatusOr<Shape> statusor = numpy::XlaShapeFromPyShape(o);
      if (!statusor.ok()) {
        PyErr_SetString(PyExc_TypeError, tensorflow::strings::StrCat("ExecutableBuildOptions.result_shape could not be created from Python shape value: ", statusor.status().ToString()).c_str());
        Py_DECREF(o);
        SWIG_fail;
      }
      build_options.set_result_layout(statusor.ValueOrDie());
    }
    Py_DECREF(o);

    $1 = &build_options;
  }
}

%ignoreall
%unignore xla;
%unignore xla::swig;
%unignore xla::swig::InitializeReplicaCount;
%unignore xla::swig::GetReplicaCount;
%unignore xla::swig::TransferToInfeedLocal;
%unignore xla::swig::TransferToInfeedLocalReplica;
%unignore xla::swig::TransferFromOutfeedLocalReplica;
%unignore xla::swig::LocalShapedBuffer;
%unignore xla::swig::LocalShapedBuffer::FromLiteral;
%unignore xla::swig::LocalShapedBuffer::ToLiteral;
%unignore xla::swig::CompiledLocalComputation;
%unignore xla::swig::CompiledLocalComputation::Execute;
%unignore xla::swig::CompiledLocalComputation::ExecuteWithShapedBuffers;
%unignore xla::swig::LocalComputation;
%unignore xla::swig::LocalComputation::Compile;
%unignore xla::swig::LocalComputation::GetReturnValueShape;
%unignore xla::swig::LocalComputationBuilder;
%unignore xla::swig::LocalComputationBuilder::LocalComputationBuilder;
%unignore xla::swig::LocalComputationBuilder::Build;
%unignore xla::swig::LocalComputationBuilder::SetOpMetadata;
%unignore xla::swig::LocalComputationBuilder::ClearOpMetadata;
%unignore xla::swig::LocalComputationBuilder::Parameter;
%unignore xla::swig::LocalComputationBuilder::GetShape;
%unignore xla::swig::LocalComputationBuilder::GetReturnValueShape;
%unignore xla::swig::LocalComputationBuilder::Infeed;
%unignore xla::swig::LocalComputationBuilder::Outfeed;
%unignore xla::swig::LocalComputationBuilder::ConstantLiteral;
%unignore xla::swig::LocalComputationBuilder::ConstantR0;
%unignore xla::swig::LocalComputationBuilder::Broadcast;
%unignore xla::swig::LocalComputationBuilder::Pad;
%unignore xla::swig::LocalComputationBuilder::Reshape;
%unignore xla::swig::LocalComputationBuilder::Collapse;
%unignore xla::swig::LocalComputationBuilder::CrossReplicaSum;
%unignore xla::swig::LocalComputationBuilder::Slice;
%unignore xla::swig::LocalComputationBuilder::SliceInDim;
%unignore xla::swig::LocalComputationBuilder::DynamicSlice;
%unignore xla::swig::LocalComputationBuilder::DynamicUpdateSlice;
%unignore xla::swig::LocalComputationBuilder::ConcatInDim;
%unignore xla::swig::LocalComputationBuilder::SelectAndScatterWithGeneralPadding;
%unignore xla::swig::LocalComputationBuilder::Select;
%unignore xla::swig::LocalComputationBuilder::Tuple;
%unignore xla::swig::LocalComputationBuilder::GetTupleElement;
%unignore xla::swig::LocalComputationBuilder::ConvertElementType;
%unignore xla::swig::LocalComputationBuilder::Call;
%unignore xla::swig::LocalComputationBuilder::Transpose;
%unignore xla::swig::LocalComputationBuilder::Rev;
%unignore xla::swig::LocalComputationBuilder::Clamp;
%unignore xla::swig::LocalComputationBuilder::Map;
%unignore xla::swig::LocalComputationBuilder::Reduce;
%unignore xla::swig::LocalComputationBuilder::ReduceWindowWithGeneralPadding;
%unignore xla::swig::LocalComputationBuilder::RngNormal;
%unignore xla::swig::LocalComputationBuilder::RngUniform;
%unignore xla::swig::LocalComputationBuilder::RngBernoulli;
%unignore xla::swig::LocalComputationBuilder::While;
%unignore xla::swig::LocalComputationBuilder::Conditional;
%unignore xla::swig::LocalComputationBuilder::IsConstant;
%unignore xla::swig::LocalComputationBuilder::Eq;
%unignore xla::swig::LocalComputationBuilder::Ne;
%unignore xla::swig::LocalComputationBuilder::Ge;
%unignore xla::swig::LocalComputationBuilder::Gt;
%unignore xla::swig::LocalComputationBuilder::Lt;
%unignore xla::swig::LocalComputationBuilder::Le;
%unignore xla::swig::LocalComputationBuilder::Dot;
%unignore xla::swig::LocalComputationBuilder::DotGeneral;
%unignore xla::swig::LocalComputationBuilder::ConvGeneralDilated;
%unignore xla::swig::LocalComputationBuilder::Add;
%unignore xla::swig::LocalComputationBuilder::Sub;
%unignore xla::swig::LocalComputationBuilder::Mul;
%unignore xla::swig::LocalComputationBuilder::Div;
%unignore xla::swig::LocalComputationBuilder::Rem;
%unignore xla::swig::LocalComputationBuilder::Max;
%unignore xla::swig::LocalComputationBuilder::Min;
%unignore xla::swig::LocalComputationBuilder::And;
%unignore xla::swig::LocalComputationBuilder::Or;
%unignore xla::swig::LocalComputationBuilder::Not;
%unignore xla::swig::LocalComputationBuilder::Abs;
%unignore xla::swig::LocalComputationBuilder::Exp;
%unignore xla::swig::LocalComputationBuilder::Floor;
%unignore xla::swig::LocalComputationBuilder::Ceil;
%unignore xla::swig::LocalComputationBuilder::Round;
%unignore xla::swig::LocalComputationBuilder::Log;
%unignore xla::swig::LocalComputationBuilder::Sign;
%unignore xla::swig::LocalComputationBuilder::Cos;
%unignore xla::swig::LocalComputationBuilder::Sin;
%unignore xla::swig::LocalComputationBuilder::Tanh;
%unignore xla::swig::LocalComputationBuilder::SqrtF32;
%unignore xla::swig::LocalComputationBuilder::SquareF32;
%unignore xla::swig::LocalComputationBuilder::Pow;
%unignore xla::swig::LocalComputationBuilder::IsFinite;
%unignore xla::swig::LocalComputationBuilder::ReciprocalF32;
%unignore xla::swig::LocalComputationBuilder::Neg;
%unignore xla::swig::LocalComputationBuilder::Sort;
%unignore xla::swig::DeleteLocalShapedBuffer;
%unignore xla::swig::DeleteLocalComputation;
%unignore xla::swig::DeleteCompiledLocalComputation;

%thread;
%include "tensorflow/compiler/xla/python/local_computation_builder.h"
%nothread;

%unignoreall
