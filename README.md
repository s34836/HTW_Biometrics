# Deep Face Verification — HTW Biometrics (Course Assignment, Part II)

This repository is the **Part II** coursework for the **Biometrics** module at **HTW Berlin**. It follows the pipeline from **Nicholas Renotte (2021): *Build a Deep Facial Recognition App from Paper to Code*** (tutorial Parts 1–8). This project extends that line of work by introducing a robust **multi-subject architecture**, advanced **quantitative evaluation**, and strict **biometric performance metrics**. The implementation was developed and tested using Python 3.11.6.

## Assignment scope (as specified)

- Reproduce the tutorial flow end-to-end (data preparation → Siamese-style network → training → verification-style use).
- Build **our own application** using **our own test/training imagery** supporting **multiple test subjects** (e.g., `igor`, `kris`, and others) rather than a single individual.
- Implement automated data separation using specific identity prefixes (`SUBJECT_NAME_uuid.jpg`) during image acquisition to prevent cross-contamination.
- Include **performance measurement** on held-out pairs with an identity-safe pairing strategy.
- Provide **helpful documentation** (this file + inline notebook comments).

## What the code does (high level)

We implement a **Siamese neural network** that embeds face patches and compares pairs via **L1 distance** and a **linear classifier + sigmoid**, utilizing **PyTorch** (with **MPS** acceleration on Apple Silicon or **CPU** fallback). 

**Training data layout** (under `data/`):

- `anchor/` — Reference face crops labeled by subject identity prefix.
- `positive/` — Same identity as the paired anchor used to evaluate genuine acceptances.
- `negative/` — Different identity samples from the academic **LFW (Labeled Faces in the Wild)** dataset used to evaluate impostor rejection.

Pairs are dynamically built from these folders ensuring no cross-identity collisions (e.g., ensuring an identity is never paired with itself as a negative match), shuffled, and split into train and test subsets.

**Application / deployment-style usage** (after training):

- Trained weights are saved as **`siamesemodelv2.pt`** in the project root.
- **`application_data/`** holds runtime assets:
  - `input_image/input_image.jpg` — Last live crop written from the camera stream during verification.
  - `verification_images/` — Gallery of reference images organized by individual identity prefixes to allow multi-subject matching.

The function `verify(model, detection_threshold, verification_threshold)` evaluates the live camera image against the distinct identity groups in the gallery. It aggregates matching consistency per person to make an accurate authorization decision rather than averaging blindly across the entire directory.

## Notebooks (recommended order)

| Notebook | Role |
|----------|------|
| **`1.ipynb`** | Environment setup: install `requirements.txt`, shared imports, device selection, data directory creation. |
| **`2.ipynb`** | **Data acquisition**: Multi-subject configuration setup; webcam capture into `anchor` / `positive` with identity-prefixed file names and a fixed **100×100** crop. |
| **`3.ipynb`** | **Full pipeline**: `preprocess`, `Dataset` / `DataLoader`, model definition, training loop, **test-set precision/recall**, weight save/load sanity check, and live batch diagnostics. |
| **`4.ipynb`** | **Lightweight app**: Loads **`siamesemodelv2.pt`** directly without training dependencies, executing the real-time multi-subject camera streams and `verify` logic. |
| **`5.ipynb`** | **Evaluation notebook**: Loads trained weights and executes an exhaustive statistical evaluation reporting the confusion matrix, ROC/DET curves, and specialized biometric error rates. |

## Dependencies

See **`requirements.txt`** (e.g. `torch`, `opencv-python`, `matplotlib`, `numpy`, `torchinfo`, `certifi`).

## How to run (short)

1. Create a virtual environment and install dependencies from **`requirements.txt`** (see `1.ipynb`).
2. Populate data folders using **`2.ipynb`** to record face profiles for all targets.
3. Run **`3.ipynb`** to train the network and export the production weights to **`siamesemodelv2.pt`**.
4. Populate your gallery inside **`application_data/verification_images/`** and launch **`4.ipynb`** or **`5.ipynb`** for operational deployment and statistical reporting.

## Performance measurement

- **Training**: Loss per epoch and validation tracking are monitored within **`3.ipynb`**.
- **Generalisation**: Pair-classification metrics (**precision** and **recall**) are calculated over the held-out test split loaders in **`3.ipynb`**.

## `5.ipynb` results (example run)

The evaluation was performed under an intentionally imbalanced configuration to simulate realistic security operational stress (bombardment by impostor attempts):
- **Sample size**: 10,000 pairs
- **Class distribution**: 9,844 negatives, 156 positives
- **Evaluation threshold**: 0.9744

Reported metrics (at threshold **0.9744**):
- Accuracy: **0.9919**
- Precision: **0.9310**
- Recall (TAR/TPR): **0.5192**
- F1-score: **0.6667**
- Specificity (TNR): **0.9994**
- Balanced Accuracy: **0.7593**
- MCC: **0.6920**
- FAR: **0.0006**
- FRR: **0.4808**
- EER: **0.0054** (achieved at an optimal threshold of **0.5889**)
- TAR @ FAR <= 0.1%: **0.5833**
- TAR @ FAR <= 1.0%: **1.0000**
- ROC-AUC: **0.9992**
- PR-AUC: **0.9241**

Interpretation:
- **Strong Discriminative Capability**: The near-perfect **ROC-AUC (0.9992)** and remarkably low **EER (0.54%)** prove that the network maps facial embeddings into highly distinct, separable clusters with minimal feature overlap.
- **Rigor and Security**: At the high evaluation threshold of `0.9744`, the system prioritizes maximum security. It locks out impostors with a near-zero False Accept Rate (**FAR = 0.06%**), meaning an unauthorized breach is practically impossible, though genuine users may need to face the camera directly to minimize false rejections (**FRR = 48.08%**).
- **Operational Optimization**: The threshold sweep shows that if the system is tuned to a standard commercial operating point of **0.5889** (the EER crossover), the overall error drops to just 0.54%. Furthermore, hitting a **100% True Accept Rate at <= 1.0% FAR** demonstrates outstanding potential for practical deployment.

## Limitations (biometrics context)

This system serves as an educational face verification prototype. Production-grade biometric software requires strict data privacy frameworks for handling face templates, hardware-backed liveness detection to counter spoofing attacks (such as printed photos or digital displays), and massive demographic validation to prevent algorithm bias.

## Academic honesty

The fundamental architecture and instructional milestones are based on the **Renotte (2021)** tutorial framework. The **PyTorch porting**, implementation of **multi-subject identity processing**, **identity-safe pair matching constraints**, **advanced biometric metrics**, and the structured evaluation pipeline are the independent coursework deliverables for HTW Berlin Biometrics Part II.