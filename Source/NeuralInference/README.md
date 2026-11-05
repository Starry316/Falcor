# Neural inference sources

Falcor builds this directory into `mpmat_inference.dll`. It contains the FP16
cooperative-vector backend, feature sampling, source appearance decoder and
model loader. No training code, Python/PyTorch, native viewer, primary traversal
or reference SDK binary is needed to build it.

The initial source snapshot is copied unchanged from MPMat's shared inference
kernels. `UPSTREAM.json` records the source revision and individual SHA256 hashes.
Local Git attributes preserve those bytes regardless of `core.autocrlf`.
The CMake integration is owned by this project. Source code is regular project
content, not a symlink, generated SDK or dependency on another checkout. When
updating the snapshot, update the shared headers and manifest together and run
the inference/rendering parity tests. The standalone MPMat renderer remains
independent.

The fixed OptiX header revision is downloaded automatically by CMake. It is
separate from Falcor's optional OptiX denoiser SDK. Generated PTX is embedded in
the DLL. The DLL uses CUDA's static runtime and the NVIDIA driver installed on
the target machine; it has no training-runtime dependencies.

The PTX embedder indexes kernel bodies at build time. Each OptiX module receives
only its selected entry and the shared declarations/functions; unused network
widths and appearance-reference entries are excluded before specialization.
This avoids compiling all 52 variants on a cold model load. PTX instructions
and layout specialization are preserved, with only CRLF newlines normalized.
Loading is still synchronous, so a short initialization pause can remain.

CUDA architectures default to `native` (CMake detects the build machine's GPUs).
Set `FALCOR_NEURAL_CUDA_ARCHITECTURES` for offline/cross builds. Compiling for a
GPU architecture does not add cooperative-vector support to an unsupported GPU.

The directory can also be configured on its own with CMake 3.24+, a compatible
C++/CUDA toolchain and Git. This isolated build uses the same source target as
Falcor and is useful for relocation and kernel tests.
