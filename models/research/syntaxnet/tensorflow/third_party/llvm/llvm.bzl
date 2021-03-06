"""This file contains BUILD extensions for generating source code from LLVM's table definition files using the TableGen tool.

See http://llvm.org/cmds/tblgen.html for more information on the TableGen
tool.
TODO(chandlerc): Currently this expresses include-based dependencies as
"sources", and has no transitive understanding due to these files not being
correctly understood by the build system.
"""

def gentbl(name, tblgen, td_file, td_srcs, tbl_outs, library = True, **kwargs):
  """gentbl() generates tabular code from a table definition file.

  Args:
    name: The name of the build rule for use in dependencies.
    tblgen: The binary used to produce the output.
    td_file: The primary table definitions file.
    td_srcs: A list of table definition files included transitively.
    tbl_outs: A list of tuples (opts, out), where each opts is a string of
      options passed to tblgen, and the out is the corresponding output file
      produced.
    library: Whether to bundle the generated files into a library.
    **kwargs: Keyword arguments to pass to subsidiary cc_library() rule.
  """
  if td_file not in td_srcs:
    td_srcs += [td_file]
  includes = []
  for (opts, out) in tbl_outs:
    outdir = out[:out.rindex("/")]
    if outdir not in includes:
      includes.append(outdir)
    rule_suffix = "_".join(opts.replace("-", "_").replace("=", "_").split(" "))
    native.genrule(
        name="%s_%s_genrule" % (name, rule_suffix),
        srcs=td_srcs,
        outs=[out],
        tools=[tblgen],
        message="Generating code from table: %s" % td_file,
        cmd=(("$(location %s) " + "-I external/llvm/include " +
              "-I external/llvm/tools/clang/include " +
              "-I $$(dirname $(location %s)) " + "%s $(location %s) -o $@") % (
                  tblgen, td_file, opts, td_file)))
  # For now, all generated files can be assumed to comprise public interfaces.
  # If this is not true, you should specify library = False
  # and list the generated '.inc' files in "srcs".
  if library:
    native.cc_library(name=name, textual_hdrs=[f for (_, f) in tbl_outs],
                      includes=includes,  **kwargs)

def llvm_target_cmake_vars(native_arch, target_triple):
  return {
      "LLVM_HOST_TRIPLE": target_triple,
      "LLVM_DEFAULT_TARGET_TRIPLE": target_triple,
      "LLVM_NATIVE_ARCH": native_arch,
  }

def _quote(s):
  """Quotes the given string for use in a shell command.

  This function double-quotes the given string (in case it contains spaces or
  other special characters) and escapes any special characters (dollar signs,
  double-quotes, and backslashes) that may be present.

  Args:
    s: The string to quote.
  Returns:
    An escaped and quoted version of the string that can be passed to a shell
    command.
  """
  return ('"' +
          s.replace("\\", "\\\\").replace("$", "\\$").replace('"', '\\"') +
          '"')

def cmake_var_string(cmake_vars):
  """Converts a dictionary to an input suitable for expand_cmake_vars.

  Ideally we would jist stringify in the expand_cmake_vars() rule, but select()
  interacts badly with genrules.

  TODO(phawkins): replace the genrule() with native rule and delete this rule.

  Args:
    cmake_vars: a dictionary with string keys and values that are convertable to
      strings.
  """
  return " ".join([_quote("{}={}".format(k, str(v)))
                   for (k, v) in cmake_vars.items()])

def expand_cmake_vars(name, src, dst, cmake_vars):
  """Expands #cmakedefine, #cmakedefine01, and CMake variables in a text file.

  Args:
    name: the name of the rule
    src: the input of the rule
    dst: the output of the rule
    cmake_vars: a string containing the CMake variables, as generated by
      cmake_var_string.
  """
  expand_cmake_vars_tool = Label("@org_tensorflow//third_party/llvm:expand_cmake_vars")
  native.genrule(
      name = name,
      srcs = [src],
      tools = [expand_cmake_vars_tool],
      outs = [dst],
      cmd = ("$(location {}) ".format(expand_cmake_vars_tool) + cmake_vars +
             "< $< > $@")
  )

