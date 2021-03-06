"""Generate zipped aar file including different variants of .so in jni folder."""

def aar_with_jni(name, android_library):
  # Generate dummy AndroidManifest.xml for dummy apk usage
  # (dummy apk is generated by <name>_dummy_app_for_so target below)
  native.genrule(
      name = name + "_binary_manifest_generator",
      outs = [name + "_generated_AndroidManifest.xml"],
      cmd = """
cat > $(OUTS) <<EOF
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
  package="dummy.package.for.so">
  <uses-sdk android:minSdkVersion="999"/>
</manifest>
EOF
""",
  )

  # Generate dummy apk including .so files and later we extract out
  # .so files and throw away the apk.
  native.android_binary(
      name = name + "_dummy_app_for_so",
      manifest = name + "_generated_AndroidManifest.xml",
      custom_package = "dummy.package.for.so",
      deps = [android_library],
      # In some platforms we don't have an Android SDK/NDK and this target
      # can't be built. We need to prevent the build system from trying to
      # use the target in that case.
      tags = ["manual"],
  )

  native.genrule(
      name = name,
      srcs = [android_library + ".aar", name + "_dummy_app_for_so_unsigned.apk"],
      outs = [name + ".aar"],
      tags = ["manual"],
      cmd = """
cp $(location {}.aar) $(location :{}.aar)
chmod +w $(location :{}.aar)
origdir=$$PWD
cd $$(mktemp -d)
unzip $$origdir/$(location :{}_dummy_app_for_so_unsigned.apk) "lib/*"
cp -r lib jni
zip -r $$origdir/$(location :{}.aar) jni/*/*.so
""".format(android_library, name, name, name, name),
  )
