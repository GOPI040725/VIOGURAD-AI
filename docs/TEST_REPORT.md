# Test report (what was actually run, and what was not)

Environment: Linux sandbox, Python 3.13, Flask 3.1, NumPy 2.5, OpenCV 5.0. **No PyTorch, no pytorchvideo, no access to
huggingface.co or PyPI.** The target platform (Windows + Python 3.12) was not available.

## Actually run and passing

`python -m unittest discover -s tests -t .` -> **61 tests, 58 passed, 3 skipped, 0 failed.**

| Area | What was exercised |
|---|---|
| Video decoding | valid AVI / MP4 / MOV (synthetic), corrupted bytes, empty file, missing file, too few frames, too long; metadata claiming more frames than exist |
| Windowing/sampling | single window, 60 s video, 1 h video capped at 24 windows, 10-frame video; RGB channel order verified (red video -> high R) |
| Aggregation | max / mean / topk / consecutive, threshold boundary, error cases |
| Pipeline | end-to-end with a **stand-in motion model (tests only)**: high-motion clip -> VIOLENCE, static clip -> NON-VIOLENCE; multi-segment long video; progress stages |
| Flask | home page + static assets, `/api/health`, upload -> background job -> result, sync `/predict`, bad extension, no file, empty file, corrupted video, too large (both limits), unknown job/route, stale upload purge, debug off |
| Cleanup | uploads directory verified empty after success, after every rejected upload, and after failures |
| Repeated use | 3 consecutive predictions -> identical result, same cached model instance reused |
| Missing model | real `ViolenceModel` with no weights -> HTTP 503 with an actionable message, nothing stored |
| Download script | success + idempotence, corrupted download rejected and removed, network error, HTTP 403, `--verify`, pinned URL/size/SHA-256 (all with a mocked network) |
| Evaluation code | confusion matrix, precision/recall/F1, Wilson interval, disjoint stratified split, folder-label discovery, threshold selection on validation only, report never claims 95 % when missed |
| Checkpoint helpers | state-dict unwrapping, `backbone.`/`module.` prefix stripping, head-layout detection (Linear / Sequential / 1-logit), stable softmax/sigmoid |
| UI (headless Chromium) | page load, model pill, file selection + preview, client-side validation message, upload -> progress -> red result page with timeline/gauge/stats, "analyze another", missing-model notice, mobile viewport (390 px, no horizontal scroll). Used the stand-in model; **JavaScript console had no errors except blocked Google Fonts** (sandbox has no external access, system fonts are the fallback) |
| Server start | `python app.py` starts, `/api/health` reports `dependency_missing` truthfully, uploads return a clear 503 |

The 3 skipped tests (loading randomly initialised X3D-M weights in both head layouts; rejecting a wrong-size file with torch present)
need PyTorch + pytorchvideo and **were not executed here**. They will run on your machine (`pytest -v`).

## NOT run / not verifiable here

* Downloading the real weights and checking the SHA-256 (network blocked).
* Loading the real checkpoint and any prediction by the real model - **so no real violent/non-violent video was classified by the real network**.
* Dependency installation on Windows / Python 3.12.
* Waitress server (not installed here; the Flask fallback was used).
* Accuracy, precision, recall, F1, false-positive/negative rates on real data.
* Mobile/Edge/Safari/Firefox rendering (only Chromium).

Passing these software tests does **not** demonstrate model accuracy.
