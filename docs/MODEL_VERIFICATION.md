# Model verification log (what was checked, and how)

Checked on 2026-10-09 by reading the public Hugging Face pages with a web-fetch tool. The build sandbox could not download
files from Hugging Face, so **no weights were downloaded or executed here**.

| Item | Finding | Source |
|---|---|---|
| Repository exists | `visionlab-ai/school-violence-detection-models`, PyTorch, task: video classification (violent / non-violent) | model card page |
| Checkpoint path | `final/final_x3d_realtime.pt` | HF tree API |
| Checkpoint size | **12,268,364 bytes** | HF tree API |
| Checkpoint SHA-256 (LFS oid) | `e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349` | HF tree API |
| Architecture | X3D-M, 16 frames, threshold 0.40, F1 0.93 (val, 400 clips); trained with dropout 0.3, lr 1e-4, mixup 0.2 | `final/MODEL_CARD.md` |
| Class mapping | index 0 = non-violent, index 1 = violent | main model card |
| Preprocessing | 16 frames from `np.linspace(0, total-1, 16)`; resize 224x224 bilinear, **no crop**; BGR->RGB; /255; mean 0.45, std 0.225; layout (1,3,T,H,W) | author's demo Space `video_utils.py` |
| Window aggregation | max over windows; alert = P(violent) >= threshold | demo Space `temporal_analyzer.py` |
| State-dict layout | keys carry a `backbone.*` prefix (wrapper `ViolenceX3D`) | `final/MODEL_CARD.md` |
| Training data | RWF-2000 (2,000 clips, 80/10/10 split), Kinetics-400 pretrained backbone | main model card |
| Licence | HF metadata: **MIT**; README text: "research and non-commercial use" - **conflicting** | model card |

## Conflicts and how they were resolved

* **Normalisation**: main card says mean 0.45 / std 0.225; `config.yaml` (as summarised by the fetch tool) says "ImageNet-style".
  The author's *running demo code* uses 0.45 / 0.225, so that is what this project uses. It is configurable in `models/model_config.json`.
* **Head layout**: the main card shows `blocks[5].proj = Linear(2048, 2)`; the final card says weights sit under a `backbone.` prefix and were trained with dropout.
  The loader therefore **detects** the head (`blocks.5.proj.weight` vs `blocks.5.proj.1.weight`), strips wrapper prefixes, and raises a clear error if required weights are missing.
  This logic is unit-tested with synthetic key maps, but **has not been run against the real checkpoint**.
* **Softmax**: pytorchvideo hub models return logits; the loader forces `head.activation = None` so softmax is applied exactly once.

## Not verified (honest list)

* Downloading the real file, and its SHA-256 matching - not possible in the sandbox.
* Loading the real checkpoint into `x3d_m` and producing real predictions.
* `pip install -r requirements.txt` on Windows / Python 3.12 (PyPI unreachable here); the version ranges are reasoned, not tested.
* That `pytorchvideo==0.1.5` imports cleanly with your installed torch/torchvision (a known torchvision incompatibility is shimmed in code, untested against the real package).
* Any accuracy figure (see ACCURACY_REPORT.md).

`python download_model.py` followed by `python check_setup.py` performs exactly these checks on your machine.
