"""Run inside windowed Mogwai via --script (also supports --fullscreen).

Checks real swapchain recreation, resizing, and timeline preservation. Wall time
is diagnostic only: compositor/driver policy can override the requested interval.
Set MPMAT_PRESENTATION_RESULT to the desired JSON result path.
"""
import json
import os
from pathlib import Path
import time
import traceback

output = Path(os.environ.get('MPMAT_PRESENTATION_RESULT', 'build/neural-validation/presentation.json'))
output.parent.mkdir(parents=True, exist_ok=True)
records = []
try:
    m.ui = False
    m.clock.pause()
    m.clock.time = 1.25
    # The Python time setter explicitly defers to the next Clock::tick().
    m.renderFrame()
    initial_time = m.clock.time
    assert abs(initial_time - 1.25) < 1e-6
    for index, enabled in enumerate((False, True, False, True, False)):
        m.vsync = enabled
        assert m.vsync == enabled
        assert m.clock.time == initial_time, 'VSync reset the scene timeline'
        if index == 3:
            m.resizeFrameBuffer(961, 541)
        for _ in range(8):
            m.renderFrame()
        start = time.perf_counter()
        for _ in range(48):
            m.renderFrame()
        elapsed = time.perf_counter() - start
        assert m.clock.time == initial_time, 'Presentation changed the paused timeline'
        records.append(dict(vsync=enabled, resized=index >= 3, frames=48,
                            meanFrameMs=elapsed*1000/48, fps=48/elapsed))
        print(records[-1], flush=True)
    output.write_text(json.dumps(dict(passed=True, records=records), indent=2), encoding='utf-8')
    exit()
except Exception:
    error = traceback.format_exc()
    print(error, flush=True)
    output.write_text(json.dumps(dict(passed=False, error=error, records=records), indent=2), encoding='utf-8')
    exit(1)
