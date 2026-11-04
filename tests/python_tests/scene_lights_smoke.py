"""Run inside Mogwai via --script to check runtime analytic-light lifecycle.

No neural assets are needed. This uses the small procedural scene supplied with
the neural demo and exercises the actual Scene GPU resource synchronization.
Set FALCOR_SCENE_LIGHTS_RESULT to select the JSON result path.
"""
import json
import os
from pathlib import Path
import traceback

from falcor import DirectionalLight, PointLight, RenderGraph, createPass, float3


root = Path(__file__).resolve().parents[2]
output = Path(os.environ.get("FALCOR_SCENE_LIGHTS_RESULT", root / "build/neural-validation/scene-lights.json"))
output.parent.mkdir(parents=True, exist_ok=True)
records = []
LIGHT_COUNT_CHANGED = 0x1000
LIGHT_FLAGS = LIGHT_COUNT_CHANGED | 0x10 | 0x20 | 0x40


def check_counts(label, total, active):
    stats = m.scene.stats
    assert len(m.scene.lights) == total, label
    assert stats["totalLightCount"] == total, label
    assert stats["activeLightCount"] == active, label
    records.append(dict(label=label, total=total, active=active))


def expect_error(action):
    try:
        action()
    except (RuntimeError, ValueError, TypeError):
        return
    raise AssertionError("Invalid light edit did not raise an exception")


try:
    m.ui = False
    m.clock.pause()
    # Mogwai advances the scene only while an active render graph exists.
    graph = RenderGraph("SceneLightsSmoke")
    graph.addPass(createPass("GBufferRaster", {"samplePattern": "Center"}), "Raster")
    graph.markOutput("Raster.posW")
    m.addGraph(graph)
    m.loadScene(str(root / "data/NeuralMaterial/NeuralMaterial.pyscene"))
    m.clock.time = 1.25
    m.renderFrame()
    assert abs(m.scene.time - 1.25) < 1e-6

    for index in reversed(range(len(m.scene.lights))):
        m.scene.removeLight(index)
    assert m.scene.refreshLighting() & LIGHT_COUNT_CHANGED
    check_counts("remove all", 0, 0)

    # Grow past the original scene's buffer capacity and verify compact active
    # ordering after both inactive insertion and deletion from the middle.
    lights = []
    for index in range(8):
        light = PointLight(f"Runtime point {index}")
        light.position = float3(index * .25, 0, 2)
        light.active = index != 3
        assert m.scene.addLight(light) == index
        lights.append(light)
        flags = m.scene.refreshLighting()
        assert flags & LIGHT_COUNT_CHANGED
        assert flags & ~LIGHT_FLAGS == 0, "Light edits dirtied scene geometry"
        check_counts(f"add {index}", index + 1, index + 1 - (index >= 3))

    expect_error(lambda: m.scene.addLight(None))
    expect_error(lambda: m.scene.addLight(lights[0]))
    expect_error(lambda: m.scene.removeLight(1000))
    check_counts("rejected edits", 8, 7)

    lights[3].active = True
    assert m.scene.refreshLighting() & LIGHT_COUNT_CHANGED
    check_counts("reactivate", 8, 8)
    lights[0].intensity = float3(2, 3, 4)
    assert m.scene.refreshLighting() & 0x20
    assert m.scene.refreshLighting() == 0, "Unchanged lighting reported changes"

    m.scene.removeLight(3)
    assert m.scene.getLight(3).name == lights[4].name
    assert m.scene.refreshLighting() & LIGHT_COUNT_CHANGED
    check_counts("remove middle", 7, 7)
    assert abs(m.scene.time - 1.25) < 1e-6, "Lighting refresh advanced animation"

    # Also exercise ordinary Scene.update without explicit refresh.
    light = DirectionalLight("Runtime directional")
    assert m.scene.addLight(light) == 7
    m.renderFrame()
    check_counts("normal update", 8, 8)
    assert m.scene.stats["directionalLightCount"] == 1
    m.clock.time = .5
    m.renderFrame()
    assert abs(m.scene.time - .5) < 1e-6, "Scene time did not follow timeline seeking"

    output.write_text(json.dumps(dict(passed=True, records=records), indent=2), encoding="utf-8")
    exit()
except Exception:
    error = traceback.format_exc()
    print(error, flush=True)
    output.write_text(json.dumps(dict(passed=False, error=error, records=records), indent=2), encoding="utf-8")
    exit(1)
