# Accuracy report

## Bottom line

**No independent accuracy measurement was possible in the build environment, so no accuracy figure is claimed for this
project. The 95 % target is neither confirmed nor refuted.**

Why: the build sandbox could not reach `huggingface.co` (egress policy) or PyPI, so the real weights could not be
downloaded, PyTorch could not be installed, and no labelled test videos were available. Everything below that is a number
is either *published by the model's authors* or *from your own original notebook*; none was measured by me.

## Figures that exist (all third-party, none independently verified)

| Source | Model | Data | Result | Independent? |
|---|---|---|---|---|
| Model authors (HF model card) | X3D-M, `final_x3d_realtime.pt` | RWF-2000 **validation** split, 400 clips | F1 **0.93** (186 TP/TN, 14 FP/FN as stated for the final models) | **No** - the validation split was used to pick checkpoint/threshold |
| Model authors | R(2+1)D-18 `final_best.pt` | same | F1 0.93 | No |
| Your notebook (`violencepredictioninvideos.ipynb`) | MoBiLSTM (MobileNetV2 + BiLSTM, 64x64) | Kaggle RLVS, random 90/10 split, 200 test videos | test accuracy **91.0 %** | Weak - random split of a dataset known to contain near-duplicate clips |

Test-split results for the new model are not published. F1 0.93 on balanced data is roughly 93 % accuracy, i.e. **below** 95 %.

## Why the new model was chosen over the old MoBiLSTM

* It is a real 3D CNN (X3D-M, pretrained on Kinetics-400, fine-tuned on RWF-2000) that models motion directly, at 224x224.
  The old pipeline uses 2D MobileNetV2 features on 64x64 frames plus a small LSTM.
* RWF-2000 is a harder, more varied benchmark than RLVS; a model that is competitive there is more likely to generalise,
  *though that is not proven here*.
* The old app also had a **train/serve colour mismatch**: the notebook read frames with `imageio` (RGB) but `app.py` fed
  OpenCV frames (BGR) to the model, a likely accuracy loss in the original deployment.

## What the pipeline does to protect accuracy

* Uses the exact preprocessing of the authors' own demo code (uniform `linspace` sampling, 224x224 bilinear resize, RGB,
  mean 0.45 / std 0.225).
* Uses the authors' recalibrated threshold (0.40) and their `max` window aggregation by default.
* Analyses long videos in 5 s overlapping windows, matching the 5 s training clips.
* Verifies the weights file by size and SHA-256 so the wrong or corrupted checkpoint cannot silently be used.

## How to get a real number (do this)

```powershell
python evaluate.py --data "C:\datasets\Real Life Violence Dataset"
```

Protocol, implemented in `evaluate.py` and unit-tested:

1. Videos found per class folder; per-class seeded split into **disjoint** validation and test sets (asserted in code).
2. Aggregation mode and threshold are chosen on **validation only**.
3. **Test** accuracy, precision, recall, F1, specificity, FP/FN counts, confusion matrix and a Wilson 95 % interval are written to
   `outputs/evaluation_report.md`. The report states "does NOT meet" the 95 % target unless the held-out test accuracy is >= 95 %.
4. The authors' fixed setting (max, 0.40) is reported alongside for comparison.

Use a dataset the model has **not** been trained on (not RWF-2000). RLVS is a reasonable choice, with the caveat that near-duplicate
clips of one source video can inflate scores and `evaluate.py` cannot detect that.

## If the measured accuracy is below 95 %

In rough order of expected payoff (these are recommendations, not verified results):

1. **Fine-tune on data resembling your deployment** (your cameras, scenes, lighting). Domain gap is the usual cause of the drop from benchmark to real use. Start from `final_x3d_realtime.pt`, hold out whole *source videos* (not clips) for validation and test.
2. **More and cleaner data**: combine RWF-2000 with other public sets (e.g. RLVS, Hockey Fights, Surveillance Fight), add hard negatives (sports, dancing, rough play), and de-duplicate near-identical clips before splitting.
3. **Temporal context / post-processing**: tune window length, overlap, `consecutive` aggregation and threshold on a validation set for the target operating point (fewer false alarms vs. fewer misses).
4. **Ensembling / larger backbone**: e.g. average X3D-M with R(2+1)D-18 (`final_best.pt`, 32 frames, threshold 0.5 per its card); the authors report the same F1 for both, so gains are uncertain and must be measured.
5. **Calibration**: fit temperature scaling on validation so the displayed percentage is closer to a true probability.

Reaching a *verified* >= 95 % on a hard, independent benchmark may require substantial data and training work; I cannot promise it.
