"""Render each model with lobe and original-appearance kernels.

Run in Mogwai with MPMAT_CONFIG naming scene, models (list of .mpmat paths),
outputDirectory and optional baselineRuntime (a previous inference DLL).
The optional baseline compares dumped inference and rendered buffers exactly.
"""
import json
import os
from pathlib import Path
import runpy
import time

config = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
root = Path(__file__).resolve().parents[2]
factory = runpy.run_path(str(root / 'scripts/NeuralMaterial.py'))['render_graph_neural_material']
directory = Path(config['outputDirectory'])
directory.mkdir(parents=True, exist_ok=True)
m.ui = False
m.clock.pause()
m.loadScene(config['scene'])
records = []
for model in config['models']:
    for reference in config.get('referenceModes', [False, True]):
        variants = [('current', '')]
        if config.get('baselineRuntime'):
            variants.insert(0, ('baseline', config['baselineRuntime']))
        captures = []
        for variant, runtime in variants:
            output = directory / (Path(model).stem + ('-reference-' if reference else '-lobes-') + variant)
            options = dict(dumpPath=str(output), captureFrame=1)
            if runtime:
                options['runtimePath'] = runtime
            if reference:
                options.update(deltaAppearanceReference=True, environmentAppearanceReference=True,
                               areaMode=2, appearanceSamplesPerFrame=1, appearanceMaxSamples=4)
            graph = factory(model, options)
            m.addGraph(graph)
            started = time.perf_counter()
            for _ in range(3):
                m.renderFrame()
            seconds = time.perf_counter() - started
            props = graph.getPass('Neural').properties
            assert not props['lastError'], props['lastError']
            assert props['shadingFrames'] >= 3, props
            assert (output / 'lobes.bin').exists(), output
            records.append(dict(model=model, reference=reference, variant=variant, seconds=seconds))
            print(records[-1], flush=True)
            captures.append(output)
            m.removeGraph(graph.name)
        if len(captures) == 2:
            baseline, current = captures
            names = {p.name for p in baseline.glob('*.bin')}
            assert names == {p.name for p in current.glob('*.bin')}, 'Capture buffers differ'
            for name in sorted(names):
                assert (baseline / name).read_bytes() == (current / name).read_bytes(), (model, reference, name)
            print('Exact buffer parity:', model, reference, flush=True)
(directory / 'results.json').write_text(json.dumps(records, indent=2))
exit()
