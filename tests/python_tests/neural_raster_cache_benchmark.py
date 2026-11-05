"""Long-running Mogwai benchmark: distinguish inference misses from Raster CPU growth.

Run with --headless --no-vsync --width 320 --height 180 --script <this file>.
Optional MPMAT_CONFIG keys: model, scene, outputDirectory, blocks, framesPerBlock.
Timing is reported, not asserted, because CPU load varies between machines.
"""
import json
import os
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[2]
config_path = os.environ.get('MPMAT_CONFIG')
config = json.loads(Path(config_path).read_text(encoding='utf-8-sig')) if config_path else {}
output = Path(config.get('outputDirectory', root/'build/neural-raster-cache-benchmark'))
output.mkdir(parents=True, exist_ok=True)
factory = runpy.run_path(str(root/'scripts/NeuralMaterial.py'))['render_graph_neural_material']
graph = factory(str(config.get('model', root/'build/neural-assets/fuzzy_mip.mpmat')),
                {'cacheInference': True, 'environment': False, 'lights': {}})
m.addGraph(graph)
m.loadScene(str(config.get('scene', root/'data/NeuralMaterial/NeuralMaterial.pyscene')))
m.ui = False
m.clock.pause()
m.profiler.enabled = True
neural = graph.getPass('Neural')
records = []
for block in range(int(config.get('blocks', 12))):
    for _ in range(int(config.get('framesPerBlock', 1000))):
        m.renderFrame()
    m.profiler.start_capture()
    for _ in range(64):
        m.renderFrame()
    timing = m.profiler.end_capture()
    props = neural.properties
    assert not props['lastError'], props['lastError']
    record = dict(block=block, inference=props['inferenceFrames'],
                  timings={key: value['stats']['mean'] for key, value in timing['events'].items()
                           if '/Raster/' in key})
    records.append(record)
    (output/'timing.json').write_text(json.dumps(records, indent=2))
    assert record['inference'] == records[0]['inference'], 'Static scene repeated inference'
    print(json.dumps(record), flush=True)
exit()
