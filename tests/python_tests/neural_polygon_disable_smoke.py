"""Mogwai GPU regression for disabled/removed polygon reference and lazy bakes."""
import json
import os
from pathlib import Path
import runpy
import struct
import zlib

root = Path(__file__).resolve().parents[2]
config = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
output = Path(config['outputDirectory'])
output.mkdir(parents=True, exist_ok=True)
factory = runpy.run_path(str(root/'scripts/NeuralMaterial.py'))['render_graph_neural_material']
light = dict(shape=0, texture=1, center=[0., 1., 0.], rotation=[30., 0., 0.], size=[2., 2.], intensity=6.)
graph = factory(config['model'], dict(environment=False, background=False, deltaLights=False,
    areaMode=2, appearanceMaxSamples=4, appearanceSamplesPerFrame=2, lights={'0': light}))
m.addGraph(graph)
m.loadScene(config['scene'])
m.ui = False
m.clock.pause()
neural = graph.getPass('Neural')
records = []

def check(label, changes, active, baseline=None):
    neural.set_properties(changes)
    for _ in range(4): m.renderFrame()
    path = output/label
    neural.set_properties({'dumpPath':str(path), 'dumpRequested':True})
    m.renderFrame()
    props = neural.properties
    assert not props['lastError'], props['lastError']
    assert props['activePolygonLights'] == active, label
    assert props['appearanceAccumulatedSamples'] == (4 if active else 0), label
    assert props['referenceBatchSamples'] == 0, label
    color = (path/'color.bin').read_bytes()
    if not active:
        assert all(max(abs(x) for x in p[:3]) == 0 for p in struct.iter_unpack('<4f', color)), label
    if baseline is not None: assert color == baseline, label
    records.append(dict(label=label, active=active, inference=int(props['inferenceFrames'])))
    return color

baseline = check('enabled', {}, 1)
check('global-off', {'areaLights':False}, 0)
check('global-on', {'areaLights':True}, 1, baseline)
check('individual-off', {'lights':{'0':dict(light, enabled=False)}}, 0)
check('individual-on', {'lights':{'0':dict(light, enabled=True)}}, 1, baseline)
check('removed', {'lights':{}}, 0)
check('readded', {'lights':{'0':light}}, 1, baseline)
missing = dict(light, texture=3, texturePath=str(output/'missing.png'), enabled=False)
check('disabled-pending-image', {'lights':{'0':missing}}, 0)
assert not neural.properties['lights']['0']['textureError'], 'Disabled light attempted image loading'
check('global-off-pending-image', {'areaLights':False, 'lights':{'0':dict(missing, enabled=True)}}, 0)
assert not neural.properties['lights']['0']['textureError'], 'Global-off light attempted image loading'
check('restore-texture', {'areaLights':True, 'lights':{'0':dict(light, enabled=True)}}, 1, baseline)

# A real source texture must survive baking and be recreated after switching to
# white or removing the light, including after reference accumulation has stopped.
def png_chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

image_path = output/'emitter.png'
image_path.write_bytes(b'\x89PNG\r\n\x1a\n'
    + png_chunk(b'IHDR', struct.pack('>2I5B', 1, 1, 8, 6, 0, 0, 0))
    + png_chunk(b'IDAT', zlib.compress(bytes((0, 64, 128, 192, 255)))) + png_chunk(b'IEND', b''))
image_light = dict(light, texture=3, texturePath=str(image_path))
image_color = check('image', {'lights':{'0':image_light}}, 1)
check('image-to-white', {'lights':{'0':dict(light, texture=0)}}, 1)
check('white-to-image', {'lights':{'0':image_light}}, 1, image_color)
check('image-removed-after-convergence', {'lights':{}}, 0)
check('image-readded', {'lights':{'0':image_light}}, 1, image_color)
assert len({r['inference'] for r in records}) == 1, 'Light edits repeated inference'
(output/'results.json').write_text(json.dumps(dict(passed=True, records=records), indent=2))
print('Polygon disable/remove/re-enable and deferred texture loading passed.', flush=True)
exit()
