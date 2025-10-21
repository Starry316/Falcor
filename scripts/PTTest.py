from falcor import *

def render_graph_PTTest():
    g = RenderGraph("PTTest")
    AccumulateSTPass = createPass("AccumulateSTPass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulateSTPass, "AccumulateSTPass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    PTTest = createPass("PTTest", {'maxBounces': 3})
    g.addPass(PTTest, "PTTest")
    # VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    # g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("AccumulateSTPass.output", "ToneMapper.src")
    # g.addEdge("VBufferRT.vbuffer", "HFTracing.vbuffer")
    # g.addEdge("VBufferRT.viewW", "HFTracing.viewW")
    g.addEdge("PTTest.color", "AccumulateSTPass.input")
    g.markOutput("ToneMapper.dst")
    return g

PTTest = render_graph_PTTest()
try: m.addGraph(PTTest)
except NameError: None
