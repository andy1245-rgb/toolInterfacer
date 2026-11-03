#!/usr/bin/env python3
"""glslc shim over glslangValidator — enough of glslc's CLI for ggml-vulkan.

This box ships glslangValidator (glslang 11.8) but not shaderc's glslc, and the
only distro glslc package is built against glibc 2.38. ggml-vulkan calls glslc
with a fixed, small argument shape (see ggml-vulkan/vulkan-shaders/
vulkan-shaders-gen.cpp:string_to_spv_func and the extension probe in its
CMakeLists), which is what this translates:

  -fshader-stage=compute --target-env=vulkan1.2 IN -o OUT [-O] [-MD -MF DEP]
  [-g] -DNAME=VAL ...

Two glslc behaviours glslangValidator lacks are emulated: the implicit
GL_GOOGLE_include_directive enable (injected after #version) and the Makefile
depfile. -O (spirv-opt) is accepted and ignored — glslangValidator emits
unoptimised SPIR-V, which is functionally equivalent.
"""
import os
import subprocess
import sys
import tempfile

STAGES = {"comp": "comp", "vert": "vert", "frag": "frag"}
INJECT = "#extension GL_GOOGLE_include_directive : require"


def main(argv):
    stage, target_env, src, out, depfile, dbg = "comp", "vulkan1.2", None, None, None, False
    defines = []
    it = iter(range(len(argv)))
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("-fshader-stage="):
            stage = STAGES.get(a.split("=", 1)[1], "comp")
        elif a.startswith("--target-env="):
            target_env = a.split("=", 1)[1]
        elif a in ("-o", "--output"):
            i += 1
            out = argv[i]
        elif a.startswith("-o") and len(a) > 2:
            out = a[2:]
        elif a == "-MF":
            i += 1
            depfile = argv[i]
        elif a == "-MD":
            depfile = depfile or ""
        elif a == "-g":
            dbg = True
        elif a.startswith("-D") or a.startswith("-I"):
            defines.append(a)
        elif a in ("-O", "-Os", "-O0") or a.startswith("--"):
            pass
        elif not a.startswith("-"):
            src = a
        i += 1

    if not src:
        sys.stderr.write("glslc-shim: no input file\n")
        return 2

    src_dir = os.path.dirname(os.path.abspath(src)) or "."
    with open(src, "r", encoding="utf-8", errors="surrogateescape") as f:
        text = f.read()

    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#version"):
            lines.insert(idx + 1, INJECT + "\n")
            break
    else:
        lines.insert(0, INJECT + "\n")

    tmpdir = tempfile.mkdtemp(prefix="glslcshim.")
    tmp_src = os.path.join(tmpdir, os.path.basename(src))
    with open(tmp_src, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write("".join(lines))

    to_stdout = out in (None, "-")
    spv = os.path.join(tmpdir, "out.spv") if to_stdout else out

    cmd = ["glslangValidator", "-V", "--target-env", target_env, "-S", stage,
           "-I" + src_dir, "-I" + tmpdir, "-o", spv]
    if dbg:
        cmd.append("-g")
    cmd.extend(defines)
    cmd.append(tmp_src)

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    diag = (proc.stdout + proc.stderr).decode("utf-8", "replace")
    if proc.returncode != 0 or "ERROR:" in diag:
        sys.stderr.write(diag)
        sys.stderr.write("glslc-shim: failed: %s\n" % " ".join(cmd))
        return proc.returncode or 1

    if to_stdout:
        with open(spv, "rb") as f:
            sys.stdout.buffer.write(f.read())
        sys.stdout.buffer.flush()

    if depfile:
        with open(depfile, "w", encoding="utf-8") as f:
            f.write("%s: %s\n" % (out if out else spv, src))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
