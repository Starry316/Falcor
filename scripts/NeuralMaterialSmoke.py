"""Headless integration check, using the same MPMAT_CONFIG as NeuralMaterial.py.

Run with frames=0 in that configuration. This verifies actual scene updates and
GPU-resident inference caching, and writes a machine-readable result.
"""
import json
import math
import os
from pathlib import Path
import runpy
import struct
from falcor import *

configuration = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
if configuration.get('frames', 0):
    raise ValueError('Smoke configuration must use frames=0')
loaded = runpy.run_path(str(Path(__file__).with_name('NeuralMaterial.py')), init_globals={'m': m})
graph = loaded['graph']
neural = graph.getPass('Neural')
m.ui = False
m.clock.pause()
records = []


def render(label, count=2):
    for _ in range(count): m.renderFrame()
    props = neural.properties
    if props['lastError']: raise RuntimeError(props['lastError'])
    record = dict(label=label, inference=int(props['inferenceFrames']), shading=int(props['shadingFrames']))
    records.append(record)
    print(record, flush=True)
    return record['inference']


initial = render('initial', 4)
assert render('static') == initial, 'Static frames repeated inference'
light = m.scene.getLight(0)
light.intensity = light.intensity * 1.5
assert render('light intensity') == initial, 'Lighting change invalidated geometry'
if m.scene.envMap:
    m.scene.envMap.rotation = float3(15, 45, 0)
    m.scene.envMap.intensity = .7
    assert render('environment rotation and intensity') == initial, 'Environment change repeated inference'
graph.getPass('ToneMapper').exposureCompensation = 2.
assert render('display exposure') == initial, 'Exposure change repeated inference'
m.scene.camera.position = m.scene.camera.position + float3(.2, 0, 0)
camera = render('camera')
assert camera > initial, 'Camera change failed to refresh inference'
assert render('camera settled') == camera
m.resizeFrameBuffer(319, 237)
resized = render('non-square resize')
assert resized > camera, 'Resize failed to refresh inference'
assert render('resize settled') == resized

# Animation remains a Falcor scene update. Reload the same demo with its optional
# animation enabled, then advance the ordinary timeline while holding the camera.
os.environ['MPMAT_ANIMATE'] = '1'
m.loadScene(configuration['scene'])
m.clock.time = 0.
animated = render('animation initial')
m.clock.time = .5
assert render('Falcor geometry animation') > animated, 'Animated geometry reused stale inference'

directory = Path(configuration.get('captureDirectory', 'build/neural-validation/lifecycle'))
directory.mkdir(parents=True, exist_ok=True)
m.frameCapture.outputDir = str(directory.resolve())
m.frameCapture.baseFilename = 'animation'
m.frameCapture.capture()

# Actual shader/interop regression: the last point light must reach CUDA with
# its per-pixel direction. A compiler side-effect/evaluation-order issue once
# left these directions zero while still producing plausible-looking RGB.
appearance_dir = directory.resolve() / 'appearance-directions'
appearance_graph = loaded['render_graph_neural_material'](configuration['model'], {
    'environment': False, 'background': False, 'areaLights': False,
    'deltaLights': True, 'deltaAppearanceReference': True,
    'dumpPath': str(appearance_dir), 'captureFrame': 1,
})
m.addGraph(appearance_graph)
for _ in range(2): m.renderFrame()
props = appearance_graph.getPass('Neural').properties
if props['lastError']: raise RuntimeError(props['lastError'])
metadata = json.loads((appearance_dir/'metadata.json').read_text())
lights = [v for _, v in sorted(metadata['activeLights'].items(), key=lambda item: int(item[0])) if v['type'] in (0, 1)]
light = lights[-1]
positions = list(struct.iter_unpack('<4f', (appearance_dir/'posW.bin').read_bytes()))
frames = list(struct.iter_unpack('<8f', (appearance_dir/'frames.bin').read_bytes()))
directions = list(struct.iter_unpack('<4f', (appearance_dir/'localDirections.bin').read_bytes()))
maximum_error, tested = 0., 0
for position, frame, actual in zip(positions, frames, directions):
    if frame[7] <= 0.: continue
    world = [light['position'][i] - position[i] for i in range(3)] if light['type'] == 0 else [-x for x in light['direction']]
    length = math.sqrt(sum(x*x for x in world))
    world = [x / length for x in world]
    t, n = frame[:3], frame[4:7]
    b = [(n[1]*t[2]-n[2]*t[1])*frame[3], (n[2]*t[0]-n[0]*t[2])*frame[3], (n[0]*t[1]-n[1]*t[0])*frame[3]]
    local = [sum(world[i]*axis[i] for i in range(3)) for axis in (t,b,n)]
    assert bool(actual[3]) == (local[2] > 0.), 'Appearance hemisphere validity mismatch'
    if local[2] <= 0.: local = [0.,0.,0.]
    maximum_error = max(maximum_error, *(abs(actual[i]-local[i]) for i in range(3)))
    tested += 1
assert tested and maximum_error < 2e-5, f'Appearance direction bridge mismatch: {maximum_error}'
(directory/'lifecycle.json').write_text(json.dumps(dict(passed=True, records=records,
    appearance_direction_pixels=tested, appearance_direction_max_error=maximum_error), indent=2))
exit()
