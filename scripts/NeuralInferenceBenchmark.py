"""Measure warmed Falcor neural mapping/inference and deferred shading.

Run this script in Mogwai with MPMAT_CONFIG naming a JSON file containing
model, scene, options and output. Defaults are 64 warmup and 96 measured frames.
Use options.cacheInference=false to measure fresh inference on every frame.
Optional width/height resize the framebuffer; otherwise Mogwai's launch size
is retained and this script does not claim an unobserved resolution.
"""
import json
import os
from pathlib import Path
import runpy

from falcor import float3


def positive_integer(configuration, name, default=None):
    value = configuration.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def snapshot(neural):
    properties = neural.properties
    if properties['lastError']:
        raise RuntimeError(properties['lastError'])
    return {
        'inferenceFrames': int(properties['inferenceFrames']),
        'shadingFrames': int(properties['shadingFrames']),
        'cacheInference': bool(properties['cacheInference']),
        'packedLobesActive': bool(properties['packedLobesActive']),
    }


configuration_text = Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig')
configuration = json.loads(configuration_text)
if not isinstance(configuration, dict):
    raise ValueError('MPMAT_CONFIG must contain a JSON object')
for key in ('model', 'scene', 'output'):
    if not isinstance(configuration.get(key), str) or not configuration[key].strip():
        raise ValueError(f'MPMAT_CONFIG requires a nonempty {key} path')
options = configuration.get('options', {})
if not isinstance(options, dict):
    raise ValueError('options must be a JSON object')
if 'cacheInference' in options and not isinstance(options['cacheInference'], bool):
    raise ValueError('options.cacheInference must be a JSON boolean')
warmup = positive_integer(configuration, 'warmup', 64)
frames = positive_integer(configuration, 'frames', 96)
dimensions = None
if 'width' in configuration or 'height' in configuration:
    dimensions = {
        'width': positive_integer(configuration, 'width'),
        'height': positive_integer(configuration, 'height'),
        'source': 'configuration.resizeFrameBuffer',
    }
for key in ('fullReceiver', 'animateDelta'):
    if key in configuration and not isinstance(configuration[key], bool):
        raise ValueError(f'{key} must be a JSON boolean')

output = Path(configuration['output'])
output.mkdir(parents=True, exist_ok=True)
# Preserve the supplied JSON literally, rather than serializing Falcor property
# wrappers or silently injecting benchmark defaults into the user's options.
(output / 'configuration.json').write_text(configuration_text, encoding='utf-8')

factory = runpy.run_path(str(Path(__file__).with_name('NeuralMaterial.py')))[
    'render_graph_neural_material'
]
graph = factory(
    configuration['model'], options, configuration.get('captureOutputs'),
    configuration.get('toneMapper'),
)
m.addGraph(graph)
m.loadScene(configuration['scene'])
m.ui = False
m.clock.pause()
m.profiler.enabled = True
if dimensions is not None:
    m.resizeFrameBuffer(dimensions['width'], dimensions['height'])
if configuration.get('fullReceiver', False):
    m.scene.camera.position = float3(0, -1.2, 2)
    m.scene.camera.target = float3(0, -1.2, 0)
    m.scene.camera.up = float3(0, 1, 0)

neural = graph.getPass('Neural')
for _ in range(warmup):
    m.renderFrame()
start = snapshot(neural)
if start['shadingFrames'] < warmup or start['inferenceFrames'] <= 0:
    raise RuntimeError(f'Neural pass did not complete warmup: {start}')
if 'cacheInference' in options and start['cacheInference'] != options['cacheInference']:
    raise RuntimeError('Neural pass did not apply options.cacheInference')

m.profiler.start_capture()
try:
    for frame in range(frames):
        if configuration.get('animateDelta', False):
            m.scene.getLight(0).intensity = float3(.3 + frame * .0001, .28, .25)
        m.renderFrame()
finally:
    timing = m.profiler.end_capture()

end = snapshot(neural)
inference_frames = end['inferenceFrames'] - start['inferenceFrames']
shading_frames = end['shadingFrames'] - start['shadingFrames']
if shading_frames != frames:
    raise RuntimeError(f'Expected {frames} measured shading frames, got {shading_frames}')
if not 0 <= inference_frames <= frames:
    raise RuntimeError(f'Invalid measured inference frame count: {inference_frames}')
if end['cacheInference'] != start['cacheInference']:
    raise RuntimeError('Inference cache setting changed during measurement')
if end['packedLobesActive'] != start['packedLobesActive']:
    raise RuntimeError('Lobe storage mode changed during measurement')
if not start['cacheInference'] and inference_frames != frames:
    raise RuntimeError(
        f'Forced inference expected on all {frames} measured frames, got {inference_frames}'
    )

diagnostics = {
    'warmupFrames': warmup,
    'measuredFrames': frames,
    'start': start,
    'end': end,
    'inferenceFramesMeasured': inference_frames,
    'shadingFramesMeasured': shading_frames,
    # Retain the previous benchmark's total-counter keys for report readers.
    'inference': end['inferenceFrames'],
    'shading': end['shadingFrames'],
    'cacheInference': end['cacheInference'],
    'packedLobesActive': end['packedLobesActive'],
    'dimensions': dimensions,
    'fullReceiver': configuration.get('fullReceiver', False),
    'animateDelta': configuration.get('animateDelta', False),
    'lastError': '',
}
(output / 'timing.json').write_text(json.dumps(timing, indent=2), encoding='utf-8')
(output / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
print(json.dumps(diagnostics, indent=2), flush=True)
exit()
