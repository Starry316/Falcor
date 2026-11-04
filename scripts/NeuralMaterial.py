"""Falcor raster -> CUDA FP16 cooperative inference -> deferred neural lighting.

Set MPMAT_MODEL / MPMAT_SCENE, or pass a JSON configuration using MPMAT_CONFIG.
This script only orchestrates Falcor. Inference never imports Python or PyTorch.
"""
import json
import os
from pathlib import Path
from falcor import *


def render_graph_neural_material(model_path, options=None, capture_outputs=None, tone_mapper=None):
    options = dict(options or {})
    options['modelPath'] = str(model_path)
    graph = RenderGraph('NeuralMaterial')
    graph.addPass(createPass('GBufferRaster', {
        'samplePattern': 'Center', 'useAlphaTest': True,
        'adjustShadingNormals': False, 'forceCullMode': True, 'cull': 'None',
    }), 'Raster')
    graph.addPass(createPass('NeuralMaterialPass', options), 'Neural')
    control = options.get('referenceControlPrefix', 'neuralReference')
    for name, domain, channel in (('AccumulateEnvironment', 'environment', 'environmentSamples'),
                                  ('AccumulateArea', 'area', 'areaSamples')):
        graph.addPass(createPass('AccumulatePass', {
            'enabled': True, 'autoReset': False, 'precisionMode': 'SingleCompensated',
            'maxFrameCount': 0, 'controlKey': control + '.' + domain,
        }), name)
        graph.addEdge('Neural.' + channel, name + '.input')
    graph.addPass(createPass('NeuralCompositePass', {'referenceControlPrefix': control}), 'Composite')
    graph.addEdge('Neural.foreground', 'Composite.foreground')
    graph.addEdge('Neural.display', 'Composite.display')
    graph.addEdge('AccumulateEnvironment.output', 'Composite.environment')
    graph.addEdge('AccumulateArea.output', 'Composite.area')
    # Match the native viewer's default gain of ten using Falcor's exposure UI.
    display = {'autoExposure': False, 'exposureCompensation': 3.321928}
    display.update(tone_mapper or {})
    graph.addPass(createPass('ToneMapper', display), 'ToneMapper')
    for channel in ('posW', 'texC', 'texGrads', 'vbuffer'):
        graph.addEdge('Raster.' + channel, 'Neural.' + channel)
    graph.addEdge('Composite.color', 'ToneMapper.src')
    graph.markOutput('ToneMapper.dst')
    for output in capture_outputs or ():
        # Migrate saved capture requests; the producer no longer emits final color.
        if output == 'Neural.color': output = 'Composite.color'
        graph.markOutput(output)
    return graph


if 'm' in globals():
    configuration = {}
    if os.environ.get('MPMAT_CONFIG'):
        configuration = json.loads(Path(os.environ['MPMAT_CONFIG']).read_text(encoding='utf-8-sig'))
    model = configuration.get('model', os.environ.get('MPMAT_MODEL', 'build/neural-assets/fuzzy_mip.mpmat'))
    scene = configuration.get('scene', os.environ.get('MPMAT_SCENE', 'Data/NeuralMaterial/NeuralMaterial.pyscene'))
    options = configuration.get('options', {})
    graph = render_graph_neural_material(model, options, configuration.get('captureOutputs'), configuration.get('toneMapper'))
    m.addGraph(graph)
    m.loadScene(scene)
    if configuration.get('frames', 0):
        m.ui = False
        m.clock.pause()
        for _ in range(int(configuration['frames'])):
            m.renderFrame()
        diagnostics = graph.getPass('Neural').properties
        if diagnostics.get('lastError'):
            raise RuntimeError(diagnostics['lastError'])
        if not diagnostics.get('shadingFrames', 0):
            raise RuntimeError('NeuralMaterialPass did not render a frame.')
        if configuration.get('captureDirectory'):
            Path(configuration['captureDirectory']).mkdir(parents=True, exist_ok=True)
            m.frameCapture.outputDir = configuration['captureDirectory']
            m.frameCapture.baseFilename = configuration.get('captureName', 'neural')
            m.frameCapture.capture()
        exit()
