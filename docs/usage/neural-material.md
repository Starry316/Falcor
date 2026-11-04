# Neural materials

`NeuralMaterialPass` integrates the MPMat material inference SDK into a Falcor
deferred render graph. Primary triangle visibility comes from `GBufferRaster`.
CUDA computes FP16 cooperative-vector lobes; Falcor computes direct lighting and
display. The first version is D3D12 only and has no shadows or indirect bounces.

## Build and run

Export the standalone SDK with MPMat's
`python -m mpmat.native_build --inference-only --sdk-dir PATH`, and export a
source-hash-matched material/head with `python -m mpmat.native`. No training
checkpoint or Python/PyTorch installation is needed at render time. Falcor's
Python scene and render-graph scripting still use its bundled interpreter.

```powershell
./build_neural.ps1 -SdkDirectory PATH_TO_SDK
./demo_neural.ps1 -Model PATH_TO_MODEL.mpmat -Environment PATH_TO_ENVIRONMENT.hdr
```

The build creates an isolated `build/windows-ninja-msvc-ci/neural-runtime/Release`
with the current Falcor binary and only the required plugins. It does not replace
the plugin manifest of an older renderer build. Other build locations can be
selected with `-BuildDirectory`.

`build_neural.ps1` verifies Ninja's MSVC header dependency tracking with a real
compile probe and repairs incorrectly decoded localized `/showIncludes`
prefixes. Use `-Clean` once when upgrading a build whose old objects recorded no
header dependencies; this avoids mixing incompatible class layouts. Normal
subsequent builds remain incremental.

With the three exported demo assets in `build/neural-assets`, use:

```powershell
./demo_neural.ps1 -Material fuzzy_mip
./demo_neural.ps1 -Material fuzzy_silk
./demo_neural.ps1 -Material shot_silk
```

Use the usual Falcor camera controls. In the render graph UI, select the Neural
pass for material, mapping, LOD and lighting controls. There is no FP32 inference
mode. The original appearance decoder is available independently for delta
lights, progressive environment integration and progressive polygon integration.
The default exposure compensation is +3.321928 EV, matching the native viewer's
gain of ten; Falcor's ToneMapper owns the display controls and tone curve.

VSync defaults off. `V` or **Global Controls > VSync (V)** changes the actual
swapchain presentation interval without resetting animation time. `-VSync`
enables it at startup, and `-Fullscreen` starts at the monitor's full resolution:

```powershell
./demo_neural.ps1 -Material fuzzy_mip -Fullscreen
```

Mogwai also accepts `--vsync` / `--no-vsync`; scripts can set `m.vsync`.
GPU profiler times measure rendering work independently of presentation waits.

## Scene and material mapping

The default scene is `Data/NeuralMaterial/NeuralMaterial.pyscene`, a Z-up plane
with a known world-to-UV scale. `-Scene` selects any compatible Falcor triangle
scene. All triangles use the selected neural asset by default. Material-ID
selection restricts the asset to one scene material; excluded foreground still
occludes the background and appears black. Mixed stock-PBR/neural shading is not
implemented in this first pass.

Mapping reconstructs the actual UV Jacobian of the current animated triangle.
By default, `flipV=true` undoes Falcor's OBJ/Assimp V flip to preserve the original
Blender/OBJ neural chart and its tangent-frame orientation. Disable it for a
scene whose UVs were authored directly in the intended Falcor convention. The
same transform is applied to the coordinate and Jacobian, not just the texture.
Thickness is in scene world units. UV repeats and source crop affect both
sampling and LOD. Automatic LOD uses GBuffer pixel-shader `ddx/ddy` before UV
wrapping; all feature planes use the same level. Non-mip sources have exactly one
feature level. Independent trained feature mips are uploaded without resampling.
Fixed fractional LOD and LOD bias are exposed separately.

Smooth/face normals, two-sided mapping, reference surface, normal-field stack
correction, parallax limit, nearest/bilinear feature filtering, periodic sampling,
and supported source synthesis remain neural controls. Falcor owns geometry,
camera, animation and exposure. The optional stack correction uses vertex
normals; a separate fitted curvature tensor is not available in this scene path.

`-Animate` enables the default scene's Falcor timeline animation. For a light
animation example, load `Data/NeuralMaterial/NeuralLightAnimation.pyscene` and bind
the polygon light to scene-light index 2. The polygon then follows that Falcor
RectLight's transform. Manual pose controls are used when no scene light is bound.

## Direct lighting

Environment, delta lights and polygon lights have separate enable controls.
Point, spot and directional lights use Falcor incident radiance/attenuation and
the neural lobe response. Default delta lighting is the cached lobe fast path.
The optional original appearance reference is FP16 CUDA with cooperative-vector
source encoding and decoding on all three demo assets. It includes the source's
learned response and coverage directly, reusing the light-independent ray code.

Environment prefiltering uses the neural normalized GGX-cos kernel, independently
convolved from the source HDR at each roughness level. Environment rotation and
intensity affect both the surface and background without repeating inference or
the convolution. The neural response already includes its cosine behavior; no
extra Smith G, Fresnel, cosine, or stock DFG term is applied. A separate background
exposure changes only the background lookup and leaves neural inference intact.

Polygon lights support rectangles, concave stars, constant emission, procedural
rainbow/sunset and loaded images. Each has position, rotation, size, intensity
and optional scene-light binding. The default LTC mode integrates the whole
polygon boundary and queries a Gaussian emission prefilter once per lobe. The
texture prefilter is normalized over the actual polygon domain. Its finite
table and source sampling have approximation errors distinct from the neural
lobe fit and LTC fit.

The polygon panel now has separate **Add star light** and **Add rectangle light**
buttons. Previously **Add polygon light** always created a star. The initial
demo light remains a star; its outline has ten edges versus a rectangle's four.

Polygon integration offers three modes: whole-light LTC, same-lobe numerical
QMC, and **Original appearance progressive reference**. The QMC mode isolates the lighting
approximation. The source mode includes the original material response and is
also suitable for examining lobe-fitting error. All modes are unoccluded direct
lighting. Emitters themselves are visible as polygon surfaces; their back sides
are black.

**Original appearance IBL reference** independently enables source-network
environment integration. The environment estimator uses a 50/50 mixture of
Falcor environment importance and cosine-hemisphere sampling. It multiplies
the decoded source response by scene-linear radiance / mixture PDF, without
another cosine. The polygon estimator samples the light's bounding rectangle,
rejects points outside the polygon and applies the emitter-area Jacobian.
Rectangles, stars, procedural textures and loaded source images are supported.

Both source references accumulate linear HDR progressively through Falcor's
`AccumulatePass` graph nodes. **Appearance samples
per frame** controls queries per enabled domain (default 1); the limit defaults
to 4096. The panel shows accumulated samples and offers a reset button. Camera,
geometry, mapping and relevant lighting changes reset history; display exposure
does not. Counter-based samples are independent of frame batching. These
references retain finite-sample noise and FP16 inference differences. Animation
continually resets the accumulation, so high sample counts are primarily for
stationary comparisons. No denoiser or temporal reuse across changed scenes is
applied.

Script options are `environmentAppearanceReference`, `areaMode=2`,
`appearanceSamplesPerFrame` (1–64) and `appearanceMaxSamples` (1–1048576).
`appearanceAccumulatedSamples` is a read-only diagnostic. Defaults preserve the
fast lobe IBL, delta and LTC paths. The existing `referenceSamples` option belongs
only to the same-lobe QMC mode.

To start with all three original-source references enabled:

```powershell
./demo_neural.ps1 -Material fuzzy_mip -OptionsJson Data/NeuralMaterial/AppearanceReference.json
```

The same preset works with `-Material fuzzy_silk` and `-Material shot_silk`.

The production LTC implementation streams the two clipping stages directly into
the spherical boundary integral, retaining only the first and previous vertices.
It preserves the exact `atan2` edge formula and the concave closed-walk semantics;
there is no new angular approximation or resolution reduction. The original
array implementation remains a numerical test oracle. A star has ten edges
versus four for a rectangle, so its integral costs more per pixel and lobe.

## Diagnostics

Pass properties can be supplied as a JSON object using `-OptionsJson`. The
`dumpPath` and `captureFrame` properties explicitly request host-readable debug
data. `metadata.json`, `rays.bin`, `frames.bin` and `lobes.bin` record the actual
raster/CUDA result for independent validation. Normal interactive frames stay
on the GPU.

```powershell
./demo_neural.ps1 -Material fuzzy_mip -Width 256 -Height 192 -Headless -Frames 2 `
  -CaptureDirectory build/neural-validation/example -OptionsJson YOUR_OPTIONS.json
```

The MPMat `python -m mpmat.falcor_validation` command compares debug lobes with
the original source/head evaluated in PyTorch. It also checks buffer layouts,
finite values, frame orthogonality, inactive pixels and feature LOD ranges.
`mpmat.falcor_mapping_validation` checks the canonical demo plane separately
against its known UV chart and actual raster derivatives. The Falcor
`scripts/NeuralMaterialSmoke.py` script checks inference caching, scene animation,
resize and the actual point-light directions delivered to CUDA. Its JSON input
uses the same `MPMAT_CONFIG` mechanism, with `frames=0`.

The Falcor GPU profiler separates mapping/cooperative inference, deferred
lighting, original-appearance reference and load-time prefilter bakes. Static
cameras can reuse inference; animated geometry, camera and mapping/LOD changes
invalidate it. Lighting-only and display-exposure changes do not. Report bake
time separately from steady-state lighting and state whether inference caching
is active when comparing frame times.

### 4K optimization check (RTX 5080, D3D12)

At 3840x2160, fuzzy_mip / three lobes, one light, and 40.07% receiver coverage,
eight warmup frames plus 24 rendered measurement frames (23 completed GPU timer
records) gave the following deferred-pass means. Environment, delta lights and
background were disabled; inference was cached. These include emitter display
and output writes, and exclude other passes and presentation waiting.

| Configuration | Before streaming clip | After |
| --- | ---: | ---: |
| Area lights disabled | 0.693 ms | 0.719 ms |
| White star, whole-light LTC | 3.251 ms | 0.986 ms |
| Textured star, whole-light LTC | 3.314 ms | 1.103 ms |
| White rectangle, whole-light LTC | — | 0.759 ms |
| Textured rectangle, whole-light LTC | — | 0.866 ms |
| White star, 64-sample numerical reference | — | 3.180 ms |

The streaming path passed 1,024 randomized GPU cases against both the previous
array clip and independent FP64 triangle integrals, including horizon clipping,
rectangles, concave stars and reversed winding. Four 384x256 FP32 output captures
(fuzzy_mip / shot_silk, white / rainbow) had maximum absolute RGB differences
at most 2.24e-8 from the previous implementation. The existing lobe/LTC/texture
prefilter fitting errors remain unchanged.

`tests/python_tests/neural_presentation_smoke.py`, run via Mogwai `--script`,
checks live VSync recreation, non-square resizing and preservation of the paused
timeline. Windowed and fullscreen D3D12 debug-layer runs passed: VSync on followed
the 144 Hz display, and off removed that cap. The test uses an empty frame to
isolate presentation; its FPS is not neural-renderer throughput.

The source-network optimization was measured with the same 4K material and
coverage, eight warmup / 24 measured frames (23 timer records). Matched SDKs
reduced forced mapping/inference from 6.597 to 4.001 ms. The updated split is
1.027 ms mapping and 2.973 ms CUDA inference/handoff. The earlier 8.09 ms
observation was from a separate run and is not the baseline for this ratio.

For two original-appearance delta lights, direction generation, decoding and
accumulation fell from 525.795 to 4.293 ms. The entire Neural pass with forced
inference fell from 541.534 to 9.251 ms. A lighting-only change takes 4.604 ms;
unchanged delta lighting reuses its result (0.780 ms Neural pass).

With source ray codes cached, the progressive reference stage takes 2.683 ms for
IBL or 2.498 ms for the star per sample per pixel, or 5.179 ms with both enabled.
These are one-sample frame costs, not the cost of a converged image. All values
exclude other graph passes, initial compilation/bakes and VSync.

`scripts/NeuralAppearanceSmoke.py` checks independent constant-response energy,
frame-batch invariance, forced-inference invariance, loaded-texture radiance and
history invalidation after environment/camera/resize/bound-light changes. It
expects a constant-response test asset and constant HDR specified in
`MPMAT_CONFIG`. This test runs actual raster, Slang, CUDA and interop paths.

### Many lights covering the framebuffer

The earlier single-light measurements used 40.07% receiver coverage. They do not
predict the cost of many lights over a full-frame material. In a 3840x2160,
fully covered fuzzy_mip test (three lobes), the previous shader reproduced
10.893 / 14.077 ms for six / eight default white stars. These are GPU frame
regions, excluding presentation waits. The scene uses a top-down camera at
(0,-1.2,2), looking at (0,-1.2,0), with the demo plane and lights at (0,0,1.8),
rotation (90,0,0), size (1.4,1.4), radiance 6. Inference remained cached.

LTC still evaluates a boundary integral per pixel, lobe and light. Two more
ten-edge stars over 4K and three lobes can add about 498 million edge evaluations.
The shader uses the original atan2 formula. Visible-emitter drawing accounted
for about 0.18 ms in the eight-star diagnostic; the bulk was surface lighting.

The updated shader conservatively classifies the light's enclosing UV rectangle
against both clipping planes. Fully rejected lights return zero; fully accepted
lights evaluate the unchanged edge integral directly. Borderline lights retain
the original clipping algorithm. LTC basis construction and LUT lookup are also
shared across lights for each lobe. No edge-angle approximation was added.

Final paired measurements used 96 warmup and 128 measured frames (127 timer
records) on RTX 5080. Absolute times varied between runs, so these comparisons
use matched before/after configurations from the longer measurement:

| Eight lights, full receiver | Previous | Updated |
| --- | ---: | ---: |
| Rectangles: Neural lighting GPU region | 7.391 ms | 5.753 ms |
| Rectangles: entire GPU frame region | 8.389 ms | 6.887 ms |
| Stars: Neural lighting GPU region | 14.358 ms | 11.540 ms |
| Stars: entire GPU frame region | 15.377 ms | 12.566 ms |

The four lighting GPU tests include 1,024 polygon cases against the independent
FP64 triangle oracle, with explicit accepted/rejected bounds, grazing cases,
rectangles, concave stars and reversed winding. Four matched 384x216 scene-linear
captures cover full-frame stars, mixed shapes/poses, textured emission and shot
silk. Maximum absolute RGB change was 1.20e-7; maximum relative L2 change was
1.79e-8. This is floating-point evaluation order, not another fitted integral.

### FP16 inference buffers and submission scheduling

The updated SDK keeps decoded cooperative lobe values in FP16 until Slang reads
them for FP32 lighting. Those values were already rounded to half in the original
decoder, so this introduces no additional quantization. Three-lobe records shrink
from 88 to 48 bytes/pixel; four-lobe records shrink from 116 to 64. Diagnostics
shows the actual storage format and size. `packedLobes=false` in JSON selects the
original float transport; an older SDK automatically falls back to that format.
Debug buffer dumps retain their original float format.

Feature maps already use RG16F/RGBA16F CUDA textures, including hardware filtering
of independent learned mips. Weights and activations remain FP16. OptiX requires
matrix and bias pointers in global memory; they are converted to its optimal
layout once at model load. The SDK now specializes immutable layout fields and
matrix offsets. `MPMAT_COOP_SPECIALIZE=0` disables specialization for A/B checks.
First-use compilation of a new layout can be slow; cached steady-frame timings
exclude it. Dynamic LOD, nearest filtering, synthesis and mapping options remain
available.

A 32-slot pinned parameter ring replaces the two-slot ring, reducing CPU waits
when several appearance decodes are submitted. GPU stream dependencies remain
intact. The separate appearance encoder is created only if actually needed;
latent lobe inference continues to reuse its source ray code.

RTX 5080 measurements at 3840x2160 used 64 warmup and 96 measured frames, with
three alternately ordered repeats. These medians of run means compare the
immediately preceding SDK with the updated SDK, with forced inference and the
same environment, two delta lights and star (256 environment-bake samples).
They exclude VSync and startup.

| Material / receiver coverage | Mapping + CUDA before | Updated | GPU frame before / updated |
| --- | ---: | ---: | ---: |
| Fuzzy mip, about 40% | 3.959 ms | 2.686 ms | 5.565 / 4.141 ms |
| Fuzzy mip, full frame | 4.798 ms | 3.831 ms | 7.924 / 6.722 ms |
| Fuzzy silk, about 40% | 4.243 ms | 3.106 ms | 5.847 / 4.567 ms |
| Shot silk, about 40% | 5.905 ms | 3.280 ms | 8.067 / 5.347 ms |

Original-appearance delta reference CPU submission falls from 6.692 to 0.150 ms;
its GPU work remains 4.286 versus 4.142 ms. These are different measurements.
Six matched captures have byte-identical rays, frames, lobes and HDR color.
The geometry wave-sharing experiment was not retained because it produced no
stable mapping improvement.

To reproduce a local benchmark from the Falcor directory:

```powershell
$env:MPMAT_CONFIG = 'data/NeuralMaterial/InferenceBenchmark.json'
./build/windows-ninja-msvc-ci/neural-runtime/Release/Mogwai.exe --headless --no-vsync --script scripts/NeuralInferenceBenchmark.py
```

The configuration sets 4K explicitly. Change `model`, `fullReceiver` and `options`
as needed. The script saves `timing.json`, `diagnostics.json` and the input JSON
in `output`, and verifies the measured inference/shading counts. To benchmark
camera motion cost, leave `cacheInference=false`; a cached static view evaluates
no MLP and mostly measures lighting. Many-light LTC is still a separate cost.

Validation passed all 1162 MPMat tests (including 29 SDK tests), four Falcor
lighting GPU tests, native CTests and the native/Falcor viewer, lifecycle and
source-reference energy checks. The SDK and deployed runtime DLL hashes match.

### Material opacity and interactive light controls

**Material opacity** fades the complete neural foreground, including the
optional substrate, into the current background. It leaves learned coverage,
cached inference and reference history unchanged. This is a final compositing
control: the raster pass still records only the nearest surface, so reducing
opacity does not reveal a second layer of scene geometry.

**Delta light controls** can add or remove point, spot and directional lights.
These are ordinary Falcor scene lights; their panels reuse Falcor's color,
power, position, direction and spot-cone controls. Point and spot lights can
orbit a world-space center. Directional lights rotate their direction, without
a positional orbit. Enabling these automatic controls takes over that light's
Falcor animation; disable automatic motion before re-enabling its scene
animation.

Manual polygon lights expose position and XYZ rotation, plus **Auto rotate**
and **Auto orbit**. Each axis has an independent enable and signed speed in
degrees per second; orbit uses the specified **Orbit center** in world units.
Environment controls expose manual XYZ rotation, intensity, tint and the same
per-axis automatic rotation. Motion is evaluated from Falcor's scene timeline,
so pausing freezes it and seeking to an earlier time reproduces the earlier
pose. Editing motion in the UI starts from the currently displayed pose.
Polygons bound to a Falcor RectLight continue to use that light's transform and
animation instead of the manual motion controls.

**Add textured light...** opens an image chooser and creates a rectangular
emitter. For an existing star or rectangle, select **Image** under **Emission
texture**, then use **Browse emission image...**, or edit the path and press
**Load emission image**. A failed replacement reports its error in that light's
panel and keeps the last successful source-image/prefilter pair. Position,
rotation, orbit, tint and intensity changes do not rebake the emission texture;
environment rotation and intensity also reuse the existing environment
prefilter. Shape or emission-source changes rebuild the polygon prefilter.

Moving a reference light changes the integral and resets the relevant
progressive history. Pause the timeline or disable automatic motion to let the
reference converge. Opacity and display exposure changes preserve that history.

Scripts can update supported live controls with
`graph.getPass("Neural").set_properties({...})`. This edits the existing pass;
`graph.updatePass("Neural", ...)` recreates it and therefore discards inference
and reference caches. `lights` supplies the complete polygon-light list, while
`analyticLights` creates or updates delta lights by name. To edit one polygon
while preserving the others, copy `neural.properties["lights"]`, change that
entry, then pass the complete list to `set_properties`.

Each polygon or analytic light can contain the following `motion` object.
The environment uses the same rotation fields under `environmentMotion`:

```json
{
  "motion": {
    "autoRotate": true,
    "rotationAxes": [1, 0, 1],
    "rotationSpeed": [15, 0, -30],
    "autoOrbit": true,
    "orbitAxes": [0, 0, 1],
    "orbitSpeed": [0, 0, 20],
    "orbitCenter": [0, 0, 0],
    "startTime": 0
  }
}
```

`startTime` is the timeline time, in seconds, at which the stored base pose
applies. Saved properties include that base pose and motion phase, plus
read-only `effectiveCenter`/`effectiveRotation` or
`effectivePosition`/`effectiveDirection` values for the current pose. For a
scripted motion edit without a jump, copy the effective pose into the base
fields, supply the complete updated motion object and set `startTime` to
`m.scene.time`. Merely replacing a motion object without these fields changes
the motion relative to its stored base and the supplied/default start time.

### Reference accumulation and Falcor's AccumulatePass

The graph in `scripts/NeuralMaterial.py` uses two instances of the built-in
`AccumulatePass`, named **AccumulateEnvironment** and **AccumulateArea**. The
Neural pass has no internal temporal reference accumulator or history buffer.
Its reference shader sums only the current frame's samples, in decoded linear
HDR. Delta-reference shading remains a deterministic sum over delta lights.

```text
GBufferRaster -> NeuralMaterialPass
                   environmentSamples -> AccumulateEnvironment --+
                   areaSamples        -> AccumulateArea ---------+-> Composite -> ToneMapper
                   foreground + display -------------------------+
```

Each sample texture stores `(RGB contribution sum, actual sample count)`.
Both accumulators use Falcor's unmodified `SingleCompensated` (FP32 Kahan)
accumulation shader. `NeuralCompositePass` divides accumulated RGB by accumulated
count, then combines deterministic lighting, material opacity, background and
emitter/debug overlays. Tone mapping follows this composition. Final scene-linear
color is **Composite.color**; the factory translates old `Neural.color` capture
requests to this output. Custom graphs must use the new edges above.

For frame sums `S_f` and attempted counts `n_f`, the stock pass produces
`(sum(S_f)/F, sum(n_f)/F)`. Their ratio is `sum(S_f)/sum(n_f)`, so changing samples
per frame and taking a partial final batch preserve per-sample weighting.
Rejected samples count as zero contributions. Material coverage stays separate
from the count. Counter-based sample identity is unchanged, although grouping
FP32 additions can produce small rounding differences from the old accumulator.

An optional dictionary `controlKey` adds producer-driven reset/hold control to
`AccumulatePass`; its accumulation math is unchanged. The graph sets
`autoReset=False` and `maxFrameCount=0`, and uses the Neural panel's **Appearance
sample limit** to stop exactly at the requested sample count. At the limit, the
stock output is held without adding duplicate frames. Increasing the limit
continues the sequence; reducing it below the current count restarts it.
Resetting either stock node requests a producer restart for the following frame.
Use the Neural reset button to restart both immediately in the next rendered
frame. The two reference domains retain their shared sample counter and reset
lifecycle. Camera, geometry, mapping and relevant lighting changes reset both;
opacity, display exposure and background exposure preserve them.

Scripts can inspect `appearanceAccumulatedSamples` and `referenceBatchSamples`
on Neural, and `accumulatedFrames` on each stock node. A frame count differs from
a sample count when there is more than one sample per frame. Multiple neural
producers in one graph need distinct `referenceControlPrefix` values, shared
with their composite and accumulator nodes. Explicit debug captures now read
the final composite and stock accumulator outputs; normal rendering has no
full-frame CPU readback.

On 2026-11-04, a Release build and `NeuralAccumulateSmoke.py`,
`NeuralControlsSmoke.py`, `NeuralMaterialSmoke.py`, `NeuralAppearanceSmoke.py`
and the four numerical lighting GPU tests passed on Windows D3D12/RTX 5080.
The new smoke checks variable batches (3+5+3), rejected-sample counts, partial
batches, cap hold/resume, both reset directions and changing sample limits.
For the same 37 samples grouped differently, final-color maximum absolute
difference was `1.86265e-8`. Against captures made before migration, the fast
path was byte-identical and the source-reference final-color difference was
`2.98023e-8`. These are correctness comparisons, not performance measurements.
The companion MPMat validation also completed: 1,162 tests passed, the source
and wheel packages built, and the train/render/viewer help commands passed.

To run the new regression, set `MPMAT_CONFIG` as for the control smoke below
and `MPMAT_ENVIRONMENT` to an HDR image, then run:

```powershell
./build/windows-ninja-msvc-ci/neural-runtime/Release/Mogwai.exe --headless --no-vsync --width 96 --height 64 --script scripts/NeuralAccumulateSmoke.py
```

Its results are written to `captureDirectory/falcor-accumulate/results.json`.

### Control regression checks (Windows D3D12)

Verified on 2026-11-04 after a clean Release build: the four numerical lighting
GPU tests, `NeuralMaterialSmoke.py`, `NeuralAppearanceSmoke.py`,
`tests/python_tests/scene_lights_smoke.py`, and `NeuralControlsSmoke.py` passed.
The control smoke checks 2,758 receiver pixels and 49 frames: opacity scales
premultiplied color/coverage, added delta lights change rendered radiance,
per-axis motion replays exactly, failed image/shape edits retain the previous
emitter, and reference histories survive opacity/display edits. Its inference
counter stays at the two initial setup/resize evaluations throughout the edits.
The tested logs contain no Slang compilation warnings. The visible UI was also
checked for point-light creation, automatic environment rotation and loading a
PNG through the native image chooser.

For the control smoke, set `MPMAT_CONFIG` to a JSON file containing `model`,
`scene`, and `captureDirectory`, using the demo receiver scene, then run:

```powershell
./build/windows-ninja-msvc-ci/neural-runtime/Release/Mogwai.exe --headless --no-vsync --width 96 --height 64 --script scripts/NeuralControlsSmoke.py
```

Results and test-only raw readbacks are saved under `captureDirectory`, including
`controls.json`. This validation changes no trained weights or inference format.
