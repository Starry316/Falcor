"""GPU integration checks for progressive source-appearance references.

Run in headless Mogwai with MPMAT_CONFIG pointing to a constant-response opaque
test asset, a constant HDR environment and the unanimated demo receiver. The
configuration supplies referenceResponse and referenceEnvironmentRadiance.
No PyTorch/NumPy is needed in the renderer. Histories are read back only here.
"""
import json
import math
import os
from pathlib import Path
import runpy
import struct
from falcor import float3

c = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
factory = runpy.run_path(str(Path(__file__).with_name('NeuralMaterial.py')))['render_graph_neural_material']
directory = Path(c['captureDirectory']).resolve()
directory.mkdir(parents=True, exist_ok=True)
m.ui = False
m.clock.pause()
samples = int(c.get('samples', 1024))
records = []


def run(batch, shape, cache=True, texture=0):
    name = f'batch{batch}_shape{shape}_cache{int(cache)}_texture{texture}'
    output = directory / name
    frames = (samples + batch - 1)//batch + 4
    options = dict(environment=True, environmentAppearanceReference=True,
        areaLights=True, areaMode=2, deltaLights=False, background=False,
        appearanceSamplesPerFrame=batch, appearanceMaxSamples=samples,
        environmentBakeSamples=16, cacheInference=cache, substrate=0.,
        dumpPath=str(output), captureFrame=frames-1,
        lights={'0': dict(shape=shape, texture=texture, center=[0.,0.,2.],
            rotation=[180.,0.,0.], size=[2.,2.], intensity=1.,
            texturePath=c.get('referenceTexture',''))})
    graph = factory(c['model'], options)
    m.addGraph(graph)
    m.loadScene(c['scene'])
    for _ in range(frames): m.renderFrame()
    props = graph.getPass('Neural').properties
    assert not props['lastError'], props['lastError']
    assert props['appearanceAccumulatedSamples'] == samples, dict(props)
    def read(name, stride):
        return list(struct.iter_unpack('<'+'f'*stride, (output/(name+'.bin')).read_bytes()))
    result = dict(name=name, graph=graph, area=read('referenceArea',4), env=read('referenceEnvironment',4),
        pos=read('posW',4), frame=read('frames',8), output=output)
    return result


def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def solid_angle(a,b,c):
    la,lb,lc = (math.sqrt(dot(v,v)) for v in (a,b,c))
    return 2*math.atan2(abs(dot(a,cross(b,c))),la*lb*lc+dot(a,b)*lc+dot(b,c)*la+dot(c,a)*lb)


baseline = run(8,1)
single = run(1,1,False)
for key in ('area','env'):
    # Falcor accumulates frame sums using compensated FP32 summation. Grouping
    # the same samples into different frame batches changes rounding only.
    maximum = max(abs(a-b) / max(1.,abs(a),abs(b))
                  for pa,pb in zip(baseline[key],single[key]) for a,b in zip(pa,pb))
    assert maximum < 3e-6, f'{key}: frame batching or forced inference changed the estimator ({maximum})'
if c.get('referenceTexture'):
    textured = run(8,1,texture=3)
    for plain, colored in zip(baseline['area'],textured['area']):
        for channel, scale in enumerate(c['referenceTextureRadiance']):
            assert abs(plain[channel]*scale-colored[channel]) < 2e-5, 'Textured emitter radiance mismatch'
rectangle = run(16,0)
response = float(c['referenceResponse'])
env_rgb = c['referenceEnvironmentRadiance']
for result, shape in ((baseline,1),(rectangle,0)):
    vertices = [(math.cos(math.pi/2+i*math.pi/5)*(1. if i%2==0 else .45),
                 -math.sin(math.pi/2+i*math.pi/5)*(1. if i%2==0 else .45),2.) for i in range(10)] if shape else [(-1,-1,2),(1,-1,2),(1,1,2),(-1,1,2)]
    area_sq = env_sq = area_ref_sq = env_ref_sq = 0.
    area_sum = area_ref_sum = env_sum = env_ref_sum = 0.
    valid = 0
    for pos,frame,area,env in zip(result['pos'],result['frame'],result['area'],result['env']):
        if frame[7] <= 0: continue
        assert area[3] == 1 and env[3] == 1, 'Source alpha was lost or multiplied twice'
        assert all(math.isfinite(x) and x >= 0 for x in area+env)
        center = tuple(a-b for a,b in zip((0,0,2),pos))
        rel = [tuple(a-b for a,b in zip(v,pos)) for v in vertices]
        omega = sum(solid_angle(center,a,b) for a,b in zip(rel,rel[1:]+rel[:1]))
        area_expected = response*omega
        for channel in range(3):
            env_expected = response*2*math.pi*env_rgb[channel]
            area_sq += (area[channel]-area_expected)**2; area_ref_sq += area_expected**2
            env_sq += (env[channel]-env_expected)**2; env_ref_sq += env_expected**2
            area_sum += area[channel]; area_ref_sum += area_expected
            env_sum += env[channel]; env_ref_sum += env_expected
        valid += 1
    errors = dict(shape=shape, pixels=valid, area_relative_l2=math.sqrt(area_sq/area_ref_sq),
        env_relative_l2=math.sqrt(env_sq/env_ref_sq), area_energy_error=abs(area_sum/area_ref_sum-1),
        env_energy_error=abs(env_sum/env_ref_sum-1))
    assert valid > 100 and errors['area_relative_l2'] < .09 and errors['env_relative_l2'] < .09, errors
    assert errors['area_energy_error'] < .015 and errors['env_energy_error'] < .015, errors
    records.append(errors)

# A directional response constant over the hemisphere integrates to 2*pi, not
# pi. This detects an erroneous extra receiver cosine in the IBL estimator.
# Reference history must reset on scene changes, while lighting never repeats
# the ray encoder and display exposure must not discard a converged history.
graph = rectangle['graph']; neural = graph.getPass('Neural')
before = neural.properties['inferenceFrames']
graph.getPass('ToneMapper').exposureCompensation = 1.
m.renderFrame()
assert neural.properties['appearanceAccumulatedSamples'] == samples
assert neural.properties['inferenceFrames'] == before
m.scene.envMap.rotation = float3(0,35,0)
m.renderFrame()
assert neural.properties['appearanceAccumulatedSamples'] == 16
assert neural.properties['inferenceFrames'] == before
m.scene.envMap.intensity = .7
m.renderFrame()
assert neural.properties['appearanceAccumulatedSamples'] == 16
assert neural.properties['inferenceFrames'] == before
m.scene.camera.position = m.scene.camera.position + float3(.2,0,0)
m.renderFrame()
assert neural.properties['appearanceAccumulatedSamples'] == 16
assert neural.properties['inferenceFrames'] > before
m.resizeFrameBuffer(80,56)
m.renderFrame()
assert neural.properties['appearanceAccumulatedSamples'] == 16

# A polygon bound to a real Falcor light follows both its intensity and animated
# transform. Neither change should invalidate the material ray code.
animated = Path(c['scene']).with_name('NeuralLightAnimation.pyscene')
if animated.is_file():
    graph = factory(c['model'],dict(environment=False,areaLights=True,areaMode=2,
        deltaLights=False,background=False,appearanceSamplesPerFrame=4,
        lights={'0':dict(sceneLight=2,shape=1)}))
    m.addGraph(graph); m.loadScene(str(animated)); m.clock.time = 0.
    neural = graph.getPass('Neural')
    for _ in range(4): m.renderFrame()
    before = neural.properties['inferenceFrames']
    rig = m.scene.getLight(2); rig.intensity = rig.intensity*1.5
    m.renderFrame()
    assert neural.properties['appearanceAccumulatedSamples'] == 4
    assert neural.properties['inferenceFrames'] == before
    m.clock.time = .5; m.renderFrame()
    assert neural.properties['appearanceAccumulatedSamples'] == 4
    assert neural.properties['inferenceFrames'] == before
    assert not neural.properties['lastError'], neural.properties['lastError']
(directory/'results.json').write_text(json.dumps(dict(passed=True,samples=samples,
    batch_numerically_equivalent=True,batch_relative_tolerance=3e-6,forced_inference_invariant=True,records=records),indent=2))
print('Original appearance energy, batching and lifecycle checks passed.',flush=True)
exit()
