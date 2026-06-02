# Deep Face Verification — HTW Biometrics (Course Assignment, Part II)

This repository is the **Part II** coursework for the **Biometrics** module at **HTW Berlin**. It follows the pipeline from **Nicholas Renotte (2021): *Build a Deep Facial Recognition App from Paper to Code*** (tutorial Parts 1–8). Primary material is the original **video series** (search the title on YouTube). This project extends that line of work with **our own image data**, **quantitative evaluation**, and **this documentation**. The implementation was developed and tested using Python 3.11.6.

## Assignment scope (as specified)

- Reproduce the tutorial flow end-to-end (data preparation → Siamese-style network → training → verification-style use).
- Build **our own application** using **our own test/training imagery** (not only the presenter’s sample set).
- Include **performance measurement** on held-out pairs.
- Provide **helpful documentation** (this file + inline notebook comments).

## What the code does (high level)

We implement a **Siamese neural network** that embeds face patches and compares pairs via **L1 distance** and a **linear classifier + sigmoid**, similar in spirit to the original Keras/TensorFlow tutorial. Here the implementation is **PyTorch** (with **MPS** on Apple Silicon or **CPU** fallback via `common_imports.py`), which keeps the same conceptual steps: paired inputs, shared weights, and a similarity score in \([0, 1]\).

**Training data layout** (under `data/`):

- `anchor/` — reference face crops  
- `positive/` — same identity as the paired anchor  
- `negative/` — different identity  

Pairs are built from these folders, shuffled, and split into **train** and **test** subsets. During training we log **loss**; on the test split we report **precision** and **recall** on model outputs, including over the **full test loader** (see `3.ipynb`).

**Application / deployment-style usage** (after training):

- Trained weights are saved as **`siamesemodelv2.pt`** in the project root.
- **`application_data/`** holds runtime artefacts:
  - `input_image/input_image.jpg` — last **live** crop written from the camera when verifying.
  - `verification_images/` — **gallery** of reference `.jpg` / `.jpeg` / `.png` / `.webp` files to compare against the live crop.

The function **`verify(model, detection_threshold, verification_threshold)`** (in `3.ipynb` / `4.ipynb`) compares the live image against every gallery image, collects similarity scores, and applies a **detection** rule (how many scores exceed `detection_threshold`) and a **verification** rule (fraction of positives vs `verification_threshold`).

## Notebooks (recommended order)

| Notebook | Role |
|----------|------|
| **`1.ipynb`** | Environment setup: install `requirements.txt`, shared imports, device selection, data directory creation. |
| **`2.ipynb`** | **Data acquisition**: optional **LFW** download/extract; webcam capture into `anchor` / `positive` with a fixed **100×100** crop (aligned with training `preprocess`). |
| **`3.ipynb`** | **Full pipeline**: `preprocess`, `Dataset` / `DataLoader`, model definition, training loop, **test-set precision/recall**, weight save/load sanity check, **`verify`**, OpenCV **Verification** window (V = save + verify, Q = quit), diagnostics. |
| **`4.ipynb`** | **Lightweight app**: loads **`siamesemodelv2.pt`** only (no training), same `verify` + camera loop — for demonstration without re-running training. |
| **`5.ipynb`** | **Evaluation notebook**: loads trained weights and reports an extended verification evaluation with confusion matrix, standard classification metrics, biometrics metrics (**FAR/FRR/EER/TAR@FAR**), and **ROC/DET** plots. |

Shared Python helpers: **`common_imports.py`** (paths, `device`, OpenCV/NumPy/PyTorch imports), **`paths.py`** / **`ANC_PATH`** for consistent roots regardless of notebook working directory.

## Dependencies

See **`requirements.txt`** (e.g. `torch`, `opencv-python`, `matplotlib`, `numpy`, `torchinfo`, `certifi` for LFW download over HTTPS).

## How to run (short)

1. Create a virtual environment, install from **`requirements.txt`** (see `1.ipynb`).
2. Populate **`data/anchor`**, **`data/positive`**, **`data/negative`** (e.g. via `2.ipynb` and/or your own crops).
3. Run **`3.ipynb`** to train, evaluate on the test split, and export **`siamesemodelv2.pt`**.
4. Add gallery images under **`application_data/verification_images/`**, then use the Verification cell (or run **`4.ipynb`** after copying weights).

## Performance measurement (where to look)

- **Training**: loss per epoch (and optional batch-level precision/recall in the training loop) in **`3.ipynb`**.
- **Generalisation**: **precision** and **recall** on the **held-out test pairs** (single batch and full `test_data` loop) in **`3.ipynb`**.

These metrics describe **pair-classification** behaviour on the collected dataset; they are **not** a substitute for formal operational evaluation (e.g. large-scale benchmarks, demographic fairness analysis, or spoof resistance).

## `5.ipynb` results (example run)

Evaluation setup used in the notebook output:
- Decision threshold: **0.9744** (selected from the EER operating point)
- Sample size: **10,000** pairs
- Class distribution: **9,966 negatives**, **34 positives** (intentionally imbalanced to stress impostor rejection)

Reported metrics (at threshold **0.9744**):
- Accuracy: **0.9412**
- Precision: **0.0518**
- Recall (TAR/TPR): **0.9412**
- F1-score: **0.0982**
- Specificity (TNR): **0.9412**
- Balanced Accuracy: **0.9412**
- MCC: **0.2133**
- FAR: **0.0588**
- FRR: **0.0588**
- EER: **0.0588** (threshold near **0.9744**)
- TAR @ FAR <= 0.1%: **0.3235**
- TAR @ FAR <= 1.0%: **0.7059**
- ROC-AUC: **0.9894**
- PR-AUC: **0.3531**

Interpretation:
- The model separates classes well overall (**high ROC-AUC**), and the selected operating threshold gives a much better security/usability balance for verification.
- Because the evaluation is strongly imbalanced, **accuracy is less informative**; FAR/FRR, TAR@FAR, and PR-AUC give a more realistic operational picture.
- We therefore selected **threshold = 0.9744**, the EER operating point, because it gives a balanced compromise where **FAR and FRR are both ~5.88%**.
- At this threshold, the model keeps high genuine acceptance (**Recall/TAR ~94.12%**) while maintaining substantially lower false acceptance (**FAR ~5.88%**).
- For deployment-style verification, threshold tuning should target an application-specific FAR (e.g., 1% or 0.1%), then report the corresponding TAR.

## Limitations (biometrics context)

This is an **educational** face **verification** demo: a small CNN on **100×100** RGB crops, user-collected data, and threshold-based decisions. Real-world biometric systems require stricter protocols, larger and more diverse data, liveness/anti-spoofing, privacy compliance, and calibrated security thresholds.

## Academic honesty

The architecture and teaching narrative follow the cited **Renotte (2021)** tutorial series; the **implementation details**, **custom dataset**, **PyTorch port**, **evaluation cells**, **`application_data` workflow**, and **documentation** are the coursework deliverables for HTW Biometrics Part II.
