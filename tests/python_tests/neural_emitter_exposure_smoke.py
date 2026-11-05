"""Mogwai regression: visible-emitter exposure must not alter surface lighting.

MPMAT_CONFIG supplies model, scene and outputDirectory.
"""
import json
import math
import os
from pathlib import Path
import runpy
import struct

config = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
root = Path(__file__).resolve().parents[2]
directory = Path(config['outputDirectory'])
directory.mkdir(parents=True, exist_ok=True)
factory = runpy.run_path(str(root / 'scripts/NeuralMaterial.py'))['render_graph_neural_material']
graph = factory(config['model'], dict(environmentBakeSamples=8, areaMode=2,
    appearanceSamplesPerFrame=2, appearanceMaxSamples=4,
    lights={'0': dict(shape=0, texture=1, center=[0., 1., 0.], rotation=[30., 0., 0.],
                      size=[2., 2.], intensity=6., color=[1., 1., 1.])}))
m.addGraph(graph)
m.loadScene(config['scene'])
m.ui = False
m.clock.pause()
neural = graph.getPass('Neural')
assert neural.properties['emitterExposure'] == -6.
for _ in range(5): m.renderFrame()
cached = neural.properties['inferenceFrames']
assert neural.properties['appearanceAccumulatedSamples'] == 4

def capture(ev):
    path = directory / str(ev)
    neural.set_properties({'emitterExposure': ev, 'dumpPath': str(path), 'dumpRequested': True})
    m.renderFrame()
    props = neural.properties
    assert not props['lastError'], props['lastError']
    assert props['inferenceFrames'] == cached
    assert props['appearanceAccumulatedSamples'] == 4
    buffers = {p.name: p.read_bytes() for p in path.glob('*.bin')}
    m.frameCapture.outputDir = str(path)
    m.frameCapture.baseFilename = 'preview'
    m.frameCapture.capture()
    return buffers

bright, dim = capture(0.), capture(-6.)
def pixels(data): return list(struct.iter_unpack('<4f', data))
before, after = pixels(bright['display.bin']), pixels(dim['display.bin'])
emitters = [i for i, p in enumerate(before) if p[3] < 0 and max(p[:3]) > 0]
assert len(emitters) > 20, 'Textured emitter is not visible'
assert len({before[i][:3] for i in emitters}) > 10, 'Emitter texture is not resolved'
for i, (a, b) in enumerate(zip(before, after)):
    if a[3] < 0:
        assert b[3] == a[3]
        for j in range(3): assert math.isclose(b[j], a[j]/64., rel_tol=1e-6, abs_tol=1e-8)
    else: assert a == b, 'Background changed'
for name in bright:
    if name not in ('display.bin', 'color.bin'):
        assert bright[name] == dim[name], ('Surface lighting or reference history changed', name)
colors_before, colors_after = pixels(bright['color.bin']), pixels(dim['color.bin'])
for i, p in enumerate(before):
    if p[3] >= 0: assert colors_before[i] == colors_after[i], 'Non-emitter output changed'
restored = capture(0.)
assert restored['color.bin'] == bright['color.bin'], 'Restoring exposure changed output'
(directory / 'results.json').write_text(json.dumps(dict(passed=True, emitterPixels=len(emitters),
    gain=1/64, inferenceFrames=int(cached), referenceSamples=4), indent=2))
print('Visible emitter exposure, surface invariance and reference history passed.', flush=True)
exit()
