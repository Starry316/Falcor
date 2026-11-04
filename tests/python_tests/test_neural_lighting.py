"""GPU numerical checks for neural-light shaders, independent of the plugin DLL.

Run with Falcor's Python 3.10 bindings and NumPy, from <runtime>/python:
  python -c "import runpy; runpy.run_path('.../test_neural_lighting.py', run_name='__main__')"
Only this test process bypasses unrelated render/import plugins. It still uses
the actual Falcor D3D12 compiler, GPU dispatch, and texture/buffer readback.
"""
import math
import os
from pathlib import Path
import unittest

os.environ['FALCOR_EMBEDDED_PYTHON'] = '1'
import falcor as f
import numpy as np


SOURCE = Path(__file__).resolve().parents[2]/'Source/RenderPasses/NeuralMaterialPass'


def create_device():
    # Keep the DLL-directory handle alive for deferred DXC loading. Falcor's
    # Python loader locates its core DLL, but DXC is loaded later by Slang.
    runtime = Path(f.__file__).resolve().parents[2]
    os.chdir(runtime)
    guard = os.add_dll_directory(str(runtime))
    return f.Device(type=f.DeviceType.D3D12), guard


def compile_compute(device, source, entry='main'):
    desc = f.ProgramDesc()
    desc.add_shader_module().add_string(source, SOURCE/'validation.slang')
    desc.cs_entry(entry)
    return f.ComputePass(device, desc)


def star_vertices():
    return np.array([(.5+(.5 if i % 2 == 0 else .225)*math.cos(math.pi/2+i*math.pi/5),
                      .5+(.5 if i % 2 == 0 else .225)*math.sin(math.pi/2+i*math.pi/5)) for i in range(10)])


def clip_reference(vertices, normal):
    result = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        za, zb = np.dot(a, normal), np.dot(b, normal)
        if (za >= 0) != (zb >= 0):
            result.append(a+(b-a)*(za/(za-zb)))
        if zb >= 0:
            result.append(b)
    return np.asarray(result).reshape(-1, 3)


def triangle_integral(vertices, inverse, normal=np.array([0., 0., 1.])):
    vertices = clip_reference(vertices, normal)
    vertices = clip_reference(vertices@inverse.T, np.array([0., 0., 1.]))
    if len(vertices) < 3:
        return 0.
    unit = vertices/np.maximum(np.linalg.norm(vertices, axis=1, keepdims=True), 1e-30)
    nxt = np.roll(unit, -1, axis=0)
    cross = np.cross(unit, nxt)
    length = np.linalg.norm(cross, axis=1)
    angle = np.arctan2(length, np.clip(np.sum(unit*nxt, axis=1), -1, 1))
    return abs(np.sum(cross[:, 2]*angle/np.maximum(length, 1e-30)))/(2*math.pi)


class TestNeuralLighting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device, cls.dll_guard = create_device()

    def test_streaming_production_integral_against_array_and_triangle_oracles(self):
        rng = np.random.default_rng(872)
        count, stride = 1024, 24
        inputs = np.zeros((count, stride, 4), np.float32)
        references = np.zeros(count)
        for j in range(count):
            vertices = star_vertices() if j % 3 else np.array([(0,0),(1,0),(1,1),(0,1)])
            if j % 2:
                vertices = vertices[::-1]
            center = rng.uniform(-2, 2, 3)
            u, v, normal, axis = rng.normal(size=(4, 3))
            normal /= np.linalg.norm(normal); axis /= np.linalg.norm(axis)
            tangent = np.cross([1,0,0] if abs(axis[2])>.95 else [0,0,1], axis)
            tangent /= np.linalg.norm(tangent)
            scale = 1. if j % 8 == 0 else np.exp(rng.uniform(np.log(.015), 0.))
            inverse = np.array([tangent/scale, np.cross(axis,tangent)/scale, axis])
            # Explicit accepted/rejected bounds and near-plane fallbacks, in
            # addition to arbitrary two-plane clipping and reversed winding.
            if j < 48:
                normal = np.array([0.,0.,1.]); inverse = np.diag([1/scale,1/scale,1.])
                center = np.array([.2,-.3,3. if j < 16 else -3. if j < 32 else (j-40)*1e-7])
                u = np.array([1.,0.,0. if j < 32 else 1.])
                v = np.array([0.,1.,0. if j < 32 else -.5])
            inputs[j,0,:3] = center; inputs[j,0,3] = len(vertices)
            inputs[j,1,:3] = u; inputs[j,2,:3] = v; inputs[j,3,:3] = normal
            inputs[j,4:7,:3] = inverse; inputs[j,8:8+len(vertices),:2] = vertices
            data = inputs[j].astype(np.float64)
            outline = data[0,:3]+(data[8:8+len(vertices),:1]-.5)*data[1,:3]+(data[8:8+len(vertices),1:2]-.5)*data[2,:3]
            for edge in range(len(vertices)):
                references[j] += triangle_integral(np.stack((data[0,:3],outline[edge],outline[(edge+1)%len(vertices)])),data[4:7,:3],data[3,:3])
        source = '''#include "NeuralLighting.slangh"
            StructuredBuffer<float4> inputData; RWStructuredBuffer<float2> outputData;
            [numthreads(64,1,1)] void main(uint3 tid:SV_DispatchThreadID) {
                uint id=tid.x;if(id>=1024)return; uint base=id*24;
                PolygonLight light={};light.center=inputData[base];light.fullU=inputData[base+1];light.fullV=inputData[base+2];
                uint count=uint(inputData[base].w);light.params.x=count;light.params.w=1;
                float3 vertices[kNeuralMaxVertices];
                for(uint i=0;i<count;++i){
                    light.uvVertices[i]=inputData[base+8+i].xy;
                    float2 uv=light.uvVertices[i]-.5f;
                    vertices[i]=light.center.xyz+uv.x*light.fullU.xyz+uv.y*light.fullV.xyz;
                }
                float3 n=inputData[base+3].xyz,r0=inputData[base+4].xyz,r1=inputData[base+5].xyz,r2=inputData[base+6].xyz;
                uint clipped=neuralClip(vertices,count,n);
                for(uint i=0;i<clipped;++i){float3 p=vertices[i];vertices[i]=float3(dot(p,r0),dot(p,r1),dot(p,r2));}
                clipped=neuralClip(vertices,clipped,float3(0,0,1));
                float reference=neuralBoundaryIntegral(vertices,clipped);
                float production=neuralStreamingIntegral(light,float3(0),n,r0,r1,r2);
                outputData[id]=float2(production,reference);
            }'''
        shader = compile_compute(self.device, source)
        src = self.device.create_structured_buffer(16, count*stride, f.ResourceBindFlags.ShaderResource)
        dst = self.device.create_structured_buffer(8, count, f.ResourceBindFlags.UnorderedAccess)
        src.from_numpy(inputs); shader.root_var['inputData']=src; shader.root_var['outputData']=dst
        shader.execute(count); self.device.wait()
        actual = dst.to_numpy().view(np.float32).reshape(count,2)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual[:,0],actual[:,1],rtol=3e-4,atol=2e-6)
        np.testing.assert_allclose(actual[:,0],references,rtol=4e-4,atol=3e-6)

    def test_production_delta_sampling_against_point_spot_directional_oracle(self):
        # This executes the shared production wrapper and real Falcor helpers.
        # In particular, the point and directional branches must not alias an
        # out-parameter struct during Slang lowering (a regression seen here).
        records, expected = [], []
        def add(kind, receiver, position, direction, intensity, opening=math.pi, penumbra=0.):
            record=np.array([list(receiver)+[kind],list(position)+[opening],
                             list(direction)+[math.cos(opening)],list(intensity)+[penumbra]],np.float32)
            records.append(record)
            receiver,position,direction,intensity=(record[i,:3].astype(np.float64) for i in range(4))
            if kind==0:
                to_light=position-receiver
                distance2=max(float(np.dot(to_light,to_light)),1e-9)
                wi=to_light/math.sqrt(distance2)
                cosine=-float(np.dot(wi,direction))
                falloff=1.
                if cosine<float(record[2,3]):
                    falloff=0.
                elif record[3,3]>0:
                    x=np.clip((float(record[1,3])-math.acos(np.clip(cosine,-1,1)))/float(record[3,3]),0,1)
                    falloff=x*x*(3-2*x)
                radiance=intensity*falloff/distance2
            else:
                wi=-direction
                radiance=intensity
            expected.append(np.r_[wi,radiance])
        for x,y in [(-2,-1),(0,0),(1.7,2.3),(-1.5,-.5)]:
            add(0,(x,y,0),(-1.5,-.5,2),(0,-1,0),(.5,.65,1))
        for angle in (0,.3,.5,.6,.69,.7,.8):
            add(0,(2*math.tan(angle),0,0),(0,0,2),(0,0,-1),(3,2,1),.7,.2)
        direction=np.array([-.3,.2,-1]); direction/=np.linalg.norm(direction)
        for receiver in [(0,0,0),(-2,3,1)]:
            add(1,receiver,(-1.5,-.5,2),direction,(.3,.28,.25))
        count=len(records)
        source='''import Scene.Lights.LightData;
            #include "NeuralDelta.slangh"
            StructuredBuffer<float4> inputData; RWStructuredBuffer<float4> outputData;
            [numthreads(64,1,1)] void main(uint3 tid:SV_DispatchThreadID) {
                uint id=tid.x;if(id>=CASE_COUNT)return;
                LightData light;
                light.type=uint(inputData[id*4].w);
                light.posW=inputData[id*4+1].xyz;light.openingAngle=inputData[id*4+1].w;
                light.dirW=inputData[id*4+2].xyz;light.cosOpeningAngle=inputData[id*4+2].w;
                light.intensity=inputData[id*4+3].xyz;light.penumbraAngle=inputData[id*4+3].w;
                AnalyticLightSample sample=neuralSampleDelta(inputData[id*4].xyz,light);
                outputData[id*2]=float4(sample.dir,0);
                outputData[id*2+1]=float4(sample.Li,0);
            }'''.replace('CASE_COUNT',str(count))
        shader=compile_compute(self.device,source)
        src=self.device.create_structured_buffer(16,count*4,f.ResourceBindFlags.ShaderResource)
        dst=self.device.create_structured_buffer(16,count*2,f.ResourceBindFlags.UnorderedAccess)
        src.from_numpy(np.asarray(records,np.float32))
        shader.root_var['inputData']=src;shader.root_var['outputData']=dst
        shader.execute(count);self.device.wait()
        actual=dst.to_numpy().view(np.float32).reshape(count,2,4)[...,:3].reshape(count,6)
        np.testing.assert_allclose(actual,np.asarray(expected),rtol=4e-5,atol=2e-7)

    def test_gaussian_volume_nodes_against_independent_double_integral(self):
        original = (SOURCE/'NeuralLightBake.cs.slang').read_text()
        start = original.index('cbuffer gLightBakeCB')
        end = original.index('};', start)+2
        flags = f.ResourceBindFlags.ShaderResource | f.ResourceBindFlags.UnorderedAccess
        for shape, kind in [('rect', 1), ('star', 0)]:
            vertices = [(0, 0), (1, 0), (1, 1), (0, 1)] if shape == 'rect' else star_vertices().tolist()
            count = len(vertices)
            vertices += [(0, 0)]*(16-count)
            literal = ','.join(f'float2({a:.9f}f,{b:.9f}f)' for a, b in vertices)
            constants = f'''static const uint3 gBakeDimensions=uint3(17,17,3);
                static const uint gBakeCells=3,gBakeTextureKind={kind},gBakeVertexCount={count},gBakeSlice=0;
                static const float gBakeLogMin=-4,gBakeLogMax=0;
                static const float2 gBakeVertices[16]={{{literal}}};'''
            shader = compile_compute(self.device, original[:start]+constants+original[end:])
            texture = self.device.create_texture(17, 17, 3, f.ResourceFormat.RGBA32Float,
                                                  mip_levels=1, bind_flags=flags)
            shader.root_var['gBakeOutput'] = texture
            shader.execute(17, 17, 3)
            self.device.wait()
            actual = texture.to_numpy().reshape(3, 17, 17, 4)[..., :3]
            self.assertTrue(np.isfinite(actual).all())
            if shape == 'star':
                np.testing.assert_allclose(actual, 1., rtol=0, atol=2e-6)
                continue
            reference = np.zeros_like(actual, dtype=np.float64)
            for z in range(3):
                for y in range(17):
                    for x in range(17):
                        uv = np.array([x, y])/16*3-1
                        outside = np.sum((uv-np.clip(uv, 0, 1))**2)
                        sigma = math.sqrt((2**(-4+2*z))**2+outside/9)
                        weights = []
                        for center in uv:
                            e = [math.erf((i/3-center)/(math.sqrt(2)*sigma)) for i in range(4)]
                            weights.append(np.diff(e)/2)
                        color, mass = np.zeros(3), 0.
                        for j in range(3):
                            for i in range(3):
                                sample = (np.floor((np.array([i+.5, j+.5])/3)*12)+.5)/12
                                phase = 4*math.pi*np.max(abs(sample-.5))
                                emission = .03+.97*(.5+.5*np.cos(phase+np.array([0, -2.0944, 2.0944])))
                                weight = weights[0][i]*weights[1][j]
                                color += weight*emission
                                mass += weight
                        reference[z, y, x] = color/mass
            # Tests only bake-node arithmetic for the same finite source grid;
            # this is not an end-to-end texture-filter/material-error tolerance.
            np.testing.assert_allclose(actual, reference, rtol=0, atol=8e-6)

    def test_concave_closed_walk_matches_independent_triangle_sum(self):
        rng = np.random.default_rng(294)
        count = 1024
        positions = rng.uniform([-3, -3, -1], [3, 3, 3], (count, 3))
        axes = rng.normal(size=(count, 3)); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        scales = np.exp(rng.uniform(math.log(.03), math.log(1.), count))
        up = np.tile([0., 0., 1.], (count, 1)); up[np.abs(axes[:, 2]) > .95] = [1., 0., 0.]
        tangents = np.cross(up, axes); tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
        bitangents = np.cross(axes, tangents)
        inverse = np.stack((tangents/scales[:, None], bitangents/scales[:, None], axes), axis=1)
        u = np.array([1.2, .3, .9]); v = np.array([-.4, 1.1, .5]); center = np.array([0., .2, 1.2])
        outline = center+(star_vertices()[:, :1]-.5)*u+(star_vertices()[:, 1:]-.5)*v
        inputs = np.zeros((count, 13, 4), np.float32)
        inputs[:, :10, :3] = outline[None]-positions[:, None]
        inputs[:, 10:, :3] = inverse
        references = np.zeros(count)
        # FP64 triangle oracle uses exactly the FP32 input coordinates consumed
        # by the GPU, isolating clipping/integration from representation error.
        for j in range(count):
            for edge in range(10):
                tri = np.stack((center-positions[j], inputs[j, edge, :3], inputs[j, (edge+1)%10, :3])).astype(np.float64)
                references[j] += triangle_integral(tri, inputs[j, 10:, :3].astype(np.float64))
        source = '''#include "NeuralLighting.slangh"
            StructuredBuffer<float4> inputData; RWStructuredBuffer<float> outputData;
            [numthreads(64,1,1)] void main(uint3 tid:SV_DispatchThreadID) {
                uint id=tid.x;if(id>=1024)return;
                float3 vertices[kNeuralMaxVertices];
                for(uint i=0;i<10;++i)vertices[i]=inputData[id*13+i].xyz;
                uint n=neuralClip(vertices,10,float3(0,0,1));
                for(uint i=0;i<n;++i){float3 p=vertices[i];vertices[i]=float3(dot(p,inputData[id*13+10].xyz),dot(p,inputData[id*13+11].xyz),dot(p,inputData[id*13+12].xyz));}
                n=neuralClip(vertices,n,float3(0,0,1));
                outputData[id]=neuralBoundaryIntegral(vertices,n);
            }'''
        shader = compile_compute(self.device, source)
        src = self.device.create_structured_buffer(16, count*13, f.ResourceBindFlags.ShaderResource)
        dst = self.device.create_structured_buffer(4, count, f.ResourceBindFlags.UnorderedAccess)
        src.from_numpy(inputs)
        shader.root_var['inputData'] = src; shader.root_var['outputData'] = dst
        shader.execute(count); self.device.wait()
        actual = dst.to_numpy().view(np.float32).reshape(-1)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, references, rtol=3e-4, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
