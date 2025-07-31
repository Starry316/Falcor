from falcor import *

def render_graph_IBL():
    g = RenderGraph("MinimalPathTracer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    NeuralMatRendering = createPass("MinimalPathTracer", {'maxBounces': 3})
    g.addPass(NeuralMatRendering, "MinimalPathTracer")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

NeuralMatRendering = render_graph_IBL()
try: m.addGraph(NeuralMatRendering)
except NameError: None
