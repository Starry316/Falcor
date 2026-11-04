"""Actual render-graph regression for Falcor AccumulatePass neural references.

MPMAT_CONFIG supplies model, scene, captureDirectory; MPMAT_ENVIRONMENT supplies
an HDR image. Uses only the embedded Python standard library and GPU readbacks.
"""
import json
import math
import os
from pathlib import Path
import runpy
import struct
import traceback

c = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
directory = Path(c['captureDirectory']).resolve() / 'falcor-accumulate'
directory.mkdir(parents=True, exist_ok=True)
records = []


def read(path):
    return list(struct.iter_unpack('<4f', path.read_bytes()))


def near(a, b, label, tolerance=3e-6):
    maximum = max(abs(x-y)/max(1., abs(x), abs(y)) for pa, pb in zip(a, b) for x, y in zip(pa, pb))
    assert maximum < tolerance, (label, maximum)
    return maximum


def snapshot(label, edit=None):
    output = directory / label
    neural.set_properties(dict(edit or {}, dumpPath=str(output), dumpRequested=True))
    m.renderFrame()
    props = neural.properties
    assert not props['lastError'], props['lastError']
    assert props['referenceAccumulator'] == 'Falcor AccumulatePass'
    assert props['inferenceFrames'] == cached, (label, 'Lighting edit repeated inference')
    record = dict(label=label, samples=int(props['appearanceAccumulatedSamples']),
                  batch=int(props['referenceBatchSamples']),
                  envFrames=int(env.properties['accumulatedFrames']),
                  areaFrames=int(area.properties['accumulatedFrames']))
    records.append(record)
    print(record, flush=True)
    result = dict(record=record, output=output)
    for name in ('color', 'referenceEnvironment', 'referenceArea', 'referenceEnvironmentAccum',
                 'referenceAreaAccum', 'environmentSamples', 'areaSamples'):
        result[name] = read(output / (name + '.bin'))
        assert all(math.isfinite(x) for pixel in result[name] for x in pixel), (label, name)
    return result


try:
    from falcor import AccumulatePass
    factory = runpy.run_path(str(Path(__file__).with_name('NeuralMaterial.py')))['render_graph_neural_material']
    options = dict(environment=True, environmentAppearanceReference=False, deltaLights=False,
                   areaLights=True, areaMode=0, background=False, environmentBakeSamples=8,
                   appearanceMaxSamples=11, appearanceSamplesPerFrame=3,
                   lights={'0': dict(shape=1, center=[8., 0., 2.], rotation=[180., 0., 0.],
                                     size=[3., 3.], intensity=3.)})
    graph = factory(c['model'], options)
    m.addGraph(graph)
    m.loadScene(c['scene'])
    m.ui = False
    m.clock.pause()
    for _ in range(4): m.renderFrame()
    neural = graph.getPass('Neural')
    env = graph.getPass('AccumulateEnvironment')
    area = graph.getPass('AccumulateArea')
    assert isinstance(env, AccumulatePass) and isinstance(area, AccumulatePass), 'Not the built-in Falcor pass'
    assert env.properties['precisionMode'] == 'SingleCompensated'
    cached = int(neural.properties['inferenceFrames'])
    idle = snapshot('idle')
    assert idle['record']['envFrames'] == idle['record']['areaFrames'] == 0

    frames = [snapshot('batch-3', {'environmentAppearanceReference': True, 'areaMode': 2})]
    frames.append(snapshot('batch-5', {'appearanceSamplesPerFrame': 5}))
    frames.append(snapshot('partial-batch-3', {'appearanceSamplesPerFrame': 7}))
    assert [f['record']['batch'] for f in frames] == [3, 5, 3]
    assert frames[-1]['record']['samples'] == 11
    assert frames[-1]['record']['envFrames'] == frames[-1]['record']['areaFrames'] == 3
    for domain, raw, accumulated in (('referenceEnvironment', 'environmentSamples', 'referenceEnvironmentAccum'),
                                      ('referenceArea', 'areaSamples', 'referenceAreaAccum')):
        expected = []
        for i in range(len(frames[0][raw])):
            for f, count in zip(frames, (3., 5., 3.)):
                assert f[raw][i][3] == count, 'Rejected samples were omitted from the denominator'
            expected.append(tuple(sum(f[raw][i][j] for f in frames)/11. for j in range(3)) + (frames[-1][domain][i][3],))
            assert abs(frames[-1][accumulated][i][3] - 11./3.) < 1e-6
        near(frames[-1][domain], expected, domain + ' variable batch weighting')

    held = snapshot('held-at-cap')
    for _ in range(5): m.renderFrame()
    held_again = snapshot('held-again', {'materialOpacity': .4, 'backgroundExposure': 2.})
    assert held_again['record']['samples'] == 11 and held_again['record']['envFrames'] == 3
    for channel in ('referenceEnvironmentAccum', 'referenceAreaAccum'):
        assert held[channel] == frames[-1][channel] == held_again[channel], 'Capped accumulator was modified'
    assert all(pixel == (0., 0., 0., 0.) for pixel in held['environmentSamples']), 'Stale batch re-exposed at sample cap'
    resumed = snapshot('resume-cap-15', {'appearanceMaxSamples': 17, 'appearanceSamplesPerFrame': 4})
    final = snapshot('resume-partial-2')
    assert resumed['record']['samples'] == 15 and final['record']['samples'] == 17
    assert final['record']['batch'] == 2 and final['record']['envFrames'] == 5

    # Reset from the actual stock node, not a Neural-only reset control. Feedback
    # must restart the producer's sequence even when it was stopped at the cap.
    env.reset()
    m.renderFrame()
    restarted = snapshot('stock-reset-restarted')
    assert restarted['record']['samples'] == 4 and restarted['record']['envFrames'] == 1
    lowered = snapshot('lower-cap', {'appearanceMaxSamples': 2})
    assert lowered['record']['samples'] == 2 and lowered['record']['batch'] == 2
    assert lowered['record']['envFrames'] == 1

    # A disabled/empty light proposal still has attempted samples and a valid
    # zero estimate. Alpha is not replaced with that count.
    empty = snapshot('empty-polygons', {'lights': {}, 'environment': False,
        'appearanceMaxSamples': 5, 'appearanceSamplesPerFrame': 3, 'resetAppearance': True})
    assert empty['record']['envFrames'] == 0 and empty['record']['areaFrames'] == 1
    assert all(pixel[:3] == (0., 0., 0.) and pixel[3] == 3. for pixel in empty['areaSamples'])
    assert all(pixel[:3] == (0., 0., 0.) for pixel in empty['referenceArea'])

    # Same 37 counter-based samples, different batches, using the same existing
    # pass and cached geometry. This tests FP32 grouping, not MC convergence.
    def sequence(label, batches):
        neural.set_properties({'environment': True, 'areaMode': 2, 'appearanceMaxSamples': 37,
            'materialOpacity': 1., 'lights': options['lights'], 'resetAppearance': True})
        index = 0
        while True:
            neural.set_properties({'appearanceSamplesPerFrame': batches[index % len(batches)]})
            m.renderFrame()
            if neural.properties['appearanceAccumulatedSamples'] == 37: break
            index += 1
        return snapshot(label)
    one = sequence('sequence-single', [1])
    variable = sequence('sequence-variable', [3, 7, 2, 11])
    errors = {channel: near(one[channel], variable[channel], channel + ' batch grouping')
              for channel in ('referenceEnvironment', 'referenceArea', 'color')}
    (directory/'results.json').write_text(json.dumps(dict(passed=True, records=records,
        groupingErrors=errors, tolerance=3e-6, inferenceFrames=cached), indent=2), encoding='utf-8')
    print('Falcor accumulator weighting, hold, reset, sample caps and sequence tests passed.', flush=True)
    exit()
except Exception:
    error = traceback.format_exc()
    print(error, flush=True)
    (directory/'results.json').write_text(json.dumps(dict(passed=False, error=error, records=records), indent=2), encoding='utf-8')
    exit(1)
