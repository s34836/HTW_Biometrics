# Deep Face Verification — HTW Biometrics (Course Assignment, Part II)

This repository is the **Part II** coursework for the **Biometrics** module at **HTW Berlin**. It follows the pipeline from **Nicholas Renotte (2021): *Build a Deep Facial Recognition App from Paper to Code*** (tutorial Parts 1–8). Primary material is the original **video series** (search the title on YouTube); a common **code companion** is [nicknochnack/SiameseNeuralNetworks](https://github.com/nicknochnack/SiameseNeuralNetworks). This project extends that line of work with **our own image data**, **quantitative evaluation**, and **this documentation**.

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

Pairs are built from these folders, shuffled, and split into **train** and **test** subsets. During training we log **loss**; on the test split we report **precision** and **recall** at a fixed decision threshold (0.5) on the model outputs, including over the **full test loader** (see `3.ipynb`).

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

## Limitations (biometrics context)

This is an **educational** face **verification** demo: a small CNN on **100×100** RGB crops, user-collected data, and threshold-based decisions. Real-world biometric systems require stricter protocols, larger and more diverse data, liveness/anti-spoofing, privacy compliance, and calibrated security thresholds.

## Academic honesty

The architecture and teaching narrative follow the cited **Renotte (2021)** tutorial series; the **implementation details**, **custom dataset**, **PyTorch port**, **evaluation cells**, **`application_data` workflow**, and **documentation** are the coursework deliverables for HTW Biometrics Part II.
