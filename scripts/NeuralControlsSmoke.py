"""Headless GPU checks for live neural material and light controls.

Run in Mogwai with MPMAT_CONFIG containing model, scene and captureDirectory.
The normal demo receiver and any exported demo material are sufficient. Tiny
PNG/HDR fixtures and raw FP32 captures use Python's standard library only.
All edits use set_properties or Scene APIs: recreating a pass would hide cache
invalidation bugs. Test-only readbacks never enter the normal display path.
"""
import json
import math
import os
from pathlib import Path
import runpy
import struct
import traceback
import zlib

from falcor import DirectionalLight, PointLight, float3


c = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
directory = Path(c['captureDirectory']).resolve()
directory.mkdir(parents=True, exist_ok=True)
records = []


def write_fixtures():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    width, height = 4, 2
    rows = b''.join(b'\0' + bytes((64, 128, 192, 255))*width for _ in range(height))
    image = directory/'constant-emitter.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 6, 0, 0, 0))
                      + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))
    environment = directory/'controls-environment.hdr'
    # RGBE scanlines shorter than eight pixels use the original uncompressed
    # representation. Unequal pixels also make rotation observable in radiance.
    environment.write_bytes(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n'
                            + bytes((32, 64, 128, 129, 128, 64, 32, 129,
                                     32, 128, 64, 129, 64, 32, 128, 129))*2)
    return image, environment


def close_vector(actual, expected, label, tolerance=2e-4):
    actual = ([float(actual.x), float(actual.y), float(actual.z)] if hasattr(actual, 'x')
              else [float(actual[i]) for i in range(3)])
    assert max(abs(a-b) for a, b in zip(actual, expected)) < tolerance, (label, actual, expected)


def render(label, count=1, expected_inference=None):
    for _ in range(count):
        m.renderFrame()
    props = neural.properties
    assert not props['lastError'], (label, props['lastError'])
    record = dict(label=label, inference=int(props['inferenceFrames']),
                  shading=int(props['shadingFrames']), samples=int(props['appearanceAccumulatedSamples']))
    if expected_inference is not None:
        assert record['inference'] == expected_inference, ('Lighting repeated inference', record, expected_inference)
    records.append(record)
    print(record, flush=True)
    return props


def edit(values):
    neural.set_properties(values)


def capture(label, values=None, expected_inference=None):
    output = directory/label
    edit(dict(values or {}, dumpPath=str(output), dumpRequested=True))
    props = render(label, expected_inference=expected_inference)
    result = {'props': props, 'directory': output}
    for name, stride in (('color', 4), ('frames', 8)):
        result[name] = list(struct.iter_unpack('<'+'f'*stride, (output/(name+'.bin')).read_bytes()))
    assert all(math.isfinite(value) for pixel in result['color'] for value in pixel), label
    result['valid'] = [i for i, frame in enumerate(result['frames']) if frame[7] > 0]
    assert len(result['valid']) > 20, (label, 'Receiver does not cover enough pixels')
    return result


def scaled_foreground(actual, baseline, scale, label):
    maximum = 0.
    for index in baseline['valid']:
        for channel in range(4):
            expected = baseline['color'][index][channel]*scale
            error = abs(actual['color'][index][channel]-expected)
            maximum = max(maximum, error)
            assert error <= 2e-7 + 3e-6*abs(expected), (label, index, channel, error, expected)
    return maximum


def check_added_light(before, after, label):
    delta_squared = reference_squared = 0.
    positive_pixels = 0
    for index in before['valid']:
        positive = False
        for channel in range(3):
            old = before['color'][index][channel]
            delta = after['color'][index][channel]-old
            assert delta >= -2e-7-2e-6*abs(old), (label, 'Added light removed radiance', index, channel, delta)
            positive |= delta > 1e-8
            delta_squared += delta*delta
            reference_squared += old*old
        positive_pixels += int(positive)
    rms = math.sqrt(delta_squared/(3*len(before['valid'])))
    relative_rms = math.sqrt(delta_squared/max(reference_squared, 1e-24))
    assert positive_pixels > max(5, len(before['valid'])//100), (label, 'New light did not reach enough material pixels')
    assert rms > 1e-8 and 1e-6 < relative_rms < 1e6, (label, rms, relative_rms)
    records.append(dict(label=label, positivePixels=positive_pixels, addedRadianceRms=rms,
                        addedRadianceRelativeRms=relative_rms))


def remove_named_light(name):
    indices = [i for i, light in enumerate(m.scene.lights) if light.name == name]
    assert len(indices) == 1, (name, indices)
    m.scene.removeLight(indices[0])


def check_history_equal(a, b):
    for name in ('referenceDirect', 'referenceEnvironment', 'referenceArea'):
        path = a['directory']/(name+'.bin')
        assert path.exists(), name
        assert path.read_bytes() == (b['directory']/(name+'.bin')).read_bytes(), ('Opacity changed raw history', name)


try:
    texture, environment = write_fixtures()
    os.environ['MPMAT_ENVIRONMENT'] = str(environment)
    os.environ['MPMAT_ANIMATE'] = '0'
    factory = runpy.run_path(str(Path(__file__).with_name('NeuralMaterial.py')))['render_graph_neural_material']
    polygon = dict(shape=0, texture=0, center=[0., 0., 2.], rotation=[180., 0., 0.],
                   size=[2., 2.], intensity=1., color=[1., 1., 1.],
                   motion=dict(autoRotate=False, autoOrbit=False))
    options = dict(environment=False, background=False, deltaLights=True, areaLights=False,
                   environmentBakeSamples=8, materialOpacity=1., substrate=0.,
                   lights={'0': polygon}, appearanceSamplesPerFrame=2, appearanceMaxSamples=8)
    graph = factory(c['model'], options)
    m.addGraph(graph)
    m.loadScene(c['scene'])
    m.resizeFrameBuffer(96, 64)
    m.ui = False
    m.clock.pause()
    m.clock.time = 0.
    neural = graph.getPass('Neural')
    render('initial', 4)
    cached = int(neural.properties['inferenceFrames'])
    baseline = capture('opacity-one', expected_inference=cached)
    assert sum(sum(baseline['color'][i][:3]) for i in baseline['valid']) > 1e-5, 'Initial material is black'
    zero = capture('opacity-zero', {'materialOpacity': 0.}, cached)
    scaled_foreground(zero, baseline, 0., 'Opacity zero')
    partial = capture('opacity-partial', {'materialOpacity': .35}, cached)
    scaled_foreground(partial, baseline, .35, 'Premultiplied opacity')
    restored = capture('opacity-restored', {'materialOpacity': 1.}, cached)
    assert restored['color'] == baseline['color'], 'Restoring opacity changed radiance'

    substrate = capture('substrate-opaque', {'substrate': .2}, cached)
    assert all(substrate['color'][i][3] == 1. for i in substrate['valid'])
    substrate_partial = capture('substrate-partial', {'materialOpacity': .25}, cached)
    scaled_foreground(substrate_partial, substrate, .25, 'Substrate opacity')
    bg_zero = capture('background-zero', {'background': True, 'substrate': 0., 'materialOpacity': 0.}, cached)
    bg_one = capture('background-one', {'materialOpacity': 1.}, cached)
    bg_partial = capture('background-partial', {'materialOpacity': .35}, cached)
    for i in bg_one['valid']:
        assert bg_partial['color'][i][3] == 1.
        for channel in range(3):
            expected = .35*bg_one['color'][i][channel]+.65*bg_zero['color'][i][channel]
            assert abs(bg_partial['color'][i][channel]-expected) < 3e-6, 'Background alpha composition'
    edit({'background': False, 'materialOpacity': 1.})

    # Runtime Scene insertion/removal must update the actual analytic-light GPU
    # list without treating the light count as changed material geometry.
    count = len(m.scene.lights)
    point = PointLight('Controls smoke point')
    point.position = float3(0, 0, 2)
    point.intensity = float3(.7, .5, .3)
    assert m.scene.addLight(point) == count
    point_image = capture('scene-add-point', expected_inference=cached)
    check_added_light(baseline, point_image, 'point-radiance')
    directional = DirectionalLight('Controls smoke directional')
    directional.direction = float3(0, 0, -1)
    directional.intensity = float3(.3)
    m.scene.addLight(directional)
    directional_image = capture('scene-add-directional', expected_inference=cached)
    check_added_light(point_image, directional_image, 'directional-radiance')
    directional.active = False
    render('scene-disable-directional', expected_inference=cached)
    point.intensity = float3(.2, .4, .6)
    render('scene-edit-point', expected_inference=cached)
    remove_named_light(directional.name)
    remove_named_light(point.name)
    removed = capture('scene-removed-lights', expected_inference=cached)
    assert len(m.scene.lights) == count
    assert removed['color'] == baseline['color'], 'Removing runtime lights changed the original scene'

    # Different enabled axes for spin and orbit, with conspicuous disabled-axis
    # speeds, catch scalar-speed implementations and accidentally coupled axes.
    spin_orbit = dict(autoRotate=True, rotationAxes=[1, 0, 1], rotationSpeed=[20., 999., -40.],
                      autoOrbit=True, orbitAxes=[0, 1, 0], orbitSpeed=[999., 90., 999.],
                      orbitCenter=[0., 0., 0.], startTime=0.)
    moving_polygon = dict(polygon, center=[1., 0., 2.], motion=spin_orbit)
    delta_motion = dict(autoRotate=True, rotationAxes=[1, 0, 0], rotationSpeed=[90., 999., 999.],
                        autoOrbit=True, orbitAxes=[0, 0, 1], orbitSpeed=[999., 999., 90.],
                        orbitCenter=[0., 0., 0.], startTime=0.)
    environment_motion = dict(autoRotate=True, rotationAxes=[0, 1, 0],
                              rotationSpeed=[999., 60., 999.], startTime=0.)
    edit({'areaLights': True, 'environment': True, 'lights': {'0': moving_polygon},
          'analyticLights': {'0': dict(name='Controls smoke spot', type='spot', position=[1., 0., 2.],
               direction=[0., 0., -1.], intensity=[.4, .3, .2], openingAngle=30., penumbraAngle=8., motion=delta_motion)},
          'environmentRotation': [10., 20., 30.], 'environmentMotion': environment_motion})
    render('motion-start', expected_inference=cached)
    spot = m.scene.getLight('Controls smoke spot')
    assert abs(spot.openingAngle-math.radians(30.)) < 1e-6
    assert abs(spot.penumbraAngle-math.radians(8.)) < 1e-6
    m.clock.time = 1.
    motion_one = capture('motion-one-second', expected_inference=cached)
    props = motion_one['props']
    close_vector(props['lights']['0']['effectiveCenter'], [2., 0., -1.], 'Polygon orbit')
    close_vector(props['lights']['0']['effectiveRotation'], [200., 0., -40.], 'Polygon spin')
    close_vector(spot.position, [0., 1., 2.], 'Delta orbit')
    close_vector(spot.direction, [0., 1., 0.], 'Delta spin')
    close_vector(props['effectiveEnvironmentRotation'], [10., 80., 30.], 'Environment spin')
    render('motion-paused', 3, cached)
    m.clock.time = .5
    half = render('motion-half-second', expected_inference=cached)
    h = math.sqrt(.5)
    close_vector(half['lights']['0']['effectiveCenter'], [3*h, 0., h], 'Polygon half orbit')
    close_vector(spot.position, [h, h, 2.], 'Delta half orbit')
    close_vector(spot.direction, [0., h, -h], 'Delta half spin')
    m.clock.time = 1.
    rewind = capture('motion-replayed', expected_inference=cached)
    assert rewind['color'] == motion_one['color'], 'Timeline seek changed deterministic motion'

    remove_named_light('Controls smoke spot')
    edit({'lights': {'0': polygon}, 'deltaLights': False, 'environment': False, 'environmentRotation': [0., 0., 0.],
          'environmentMotion': {'autoRotate': False}})
    # Exactly one successful image bake; changing only a path to a missing file
    # must retain the previous GPU source and prefilter, while exposing an error.
    image_polygon = dict(polygon, texture=3, texturePath=str(texture))
    image = capture('image-light', {'lights': {'0': image_polygon}}, cached)
    assert not image['props']['lights']['0']['textureError']
    assert sum(sum(image['color'][i][:3]) for i in image['valid']) > 1e-6
    missing = dict(image_polygon, texturePath=str(directory/'does-not-exist.png'))
    failed = capture('image-load-failed', {'lights': {'0': missing}}, cached)
    assert failed['props']['lights']['0']['textureError'], 'Missing image was not reported'
    assert failed['color'] == image['color'], 'Failed image replacement lost the active emitter texture'
    failed_shape = capture('image-and-shape-load-failed',
                           {'lights': {'0': dict(missing, shape=1)}}, cached)
    assert failed_shape['props']['lights']['0']['textureError'], 'Missing image with a changed shape was not reported'
    assert failed_shape['props']['lights']['0']['shape'] == image['props']['lights']['0']['shape'], \
        'Failed texture replacement kept a shape incompatible with the retained prefilter'
    assert failed_shape['color'] == image['color'], 'Texture rollback changed the active polygon outline or emission'
    render('image-failure-settled', 2, cached)

    # Reference samples remain in scene-linear radiance. Opacity and display
    # changes operate after raw history, whereas lighting edits restart it.
    # Place this emitter outside the demo camera's view while still above the
    # receiver. Directly visible emitters have their own coverage and correctly
    # stay opaque when material opacity changes; they must not enter this check.
    reference_polygon = dict(polygon, center=[8., 0., 2.], size=[1., 1.])
    edit({'lights': {'0': reference_polygon}, 'environment': True, 'environmentAppearanceReference': True,
          'areaMode': 2, 'deltaLights': True, 'deltaAppearanceReference': True,
          'materialOpacity': 1., 'resetAppearance': True})
    render('reference-converged', 5, cached)
    reference = capture('reference-opacity-one', expected_inference=cached)
    assert reference['props']['appearanceAccumulatedSamples'] == 8
    reference_zero = capture('reference-opacity-zero', {'materialOpacity': 0.}, cached)
    scaled_foreground(reference_zero, reference, 0., 'Appearance opacity zero / out-of-view emitter')
    check_history_equal(reference, reference_zero)
    reference_partial = capture('reference-opacity-partial', {'materialOpacity': .35}, cached)
    scaled_foreground(reference_partial, reference, .35, 'Appearance opacity')
    check_history_equal(reference, reference_partial)
    graph.getPass('ToneMapper').exposureCompensation = 1.
    display = capture('reference-display-only', expected_inference=cached)
    assert display['props']['appearanceAccumulatedSamples'] == 8
    check_history_equal(reference, display)
    edit({'lights': {'0': dict(reference_polygon, intensity=1.5)}})
    assert render('reference-polygon-reset', expected_inference=cached)['appearanceAccumulatedSamples'] == 2
    render('reference-settle', 4, cached)
    edit({'environmentRotation': [0., 25., 0.]})
    assert render('reference-environment-reset', expected_inference=cached)['appearanceAccumulatedSamples'] == 2
    render('reference-final-settle', 4, cached)
    final = neural.properties
    assert final['appearanceAccumulatedSamples'] == 8
    assert final['inferenceFrames'] == cached
    (directory/'controls.json').write_text(json.dumps(dict(passed=True, inferenceFrames=cached,
        checkedPixels=len(baseline['valid']), records=records), indent=2), encoding='utf-8')
    print('Live opacity, light controls, texture rollback, motion and cache checks passed.', flush=True)
    exit()
except Exception:
    error = traceback.format_exc()
    print(error, flush=True)
    (directory/'controls.json').write_text(json.dumps(dict(passed=False, error=error, records=records), indent=2), encoding='utf-8')
    exit(1)
