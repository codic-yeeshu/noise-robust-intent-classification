# AWS SageMaker Quick-Start & Presentation Guide

## Files in This Folder
- **`NLP_Final_Project_Robust_Intent_Classification_SageMaker.ipynb`** — Complete, self-contained final project notebook covering **Milestones M1–M6** (Industry Track + Research-grade Ablation & Failure Analysis).

---

## Step-by-Step: Running on AWS SageMaker

1. **Open AWS SageMaker** (SageMaker Studio, Studio Lab, or Notebook Instance).
2. **Choose an Instance & Kernel:**
   - **Recommended GPU instance:** `ml.g4dn.xlarge` (NVIDIA T4, 16 GB VRAM) or `ml.g5.xlarge` (A10G, 24 GB VRAM) — full run takes **~10–15 minutes**.
   - **Kernel:** `Python 3 (PyTorch GPU)` (or any standard PyTorch kernel).
   - *(CPU-only fallback: set `FAST_DEMO_MODE = True` in Cell 3 to run a 2–3 minute end-to-end smoke test.)*
3. **Upload the Notebook:** Upload **only** `NLP_Final_Project_Robust_Intent_Classification_SageMaker.ipynb`. No external datasets or scripts need to be uploaded.
4. **Run Everything:** Click **Run → Run All Cells**.
   - Cell 2 installs `transformers`, `datasets`, `scikit-learn`, `ipywidgets`, etc., without overwriting SageMaker's CUDA PyTorch.
   - Cell 5 automatically downloads the **Bitext 27-Intent Customer Support Dataset** (~26.8K utterances) from HuggingFace (with a direct CSV fallback URL).
5. **Crash-Safe Auto-Checkpointing:**
   - Every **BiLSTM** and **DistilBERT** model atomically saves a checkpoint to `outputs/checkpoints/` **after every epoch** (including model weights, optimizer, scheduler, AMP scaler, best validation state, and training history).
   - If SageMaker crashes, times out, or disconnects mid-training, just click **Run → Run All Cells** again: already-completed models load instantly and the interrupted model resumes seamlessly from the next epoch. (To force a full retrain from scratch, set `CONFIG["resume_from_checkpoints"] = False` in Cell 3.)
6. **Outputs Generated Automatically (`./outputs/`):**
   - `outputs/plots/` — 5 presentation-ready high-DPI PNG charts (EDA, Robustness Heatmap & Ablation, Paradigm Comparison & Severity Sweep, 27×27 Confusion Matrices, Calibration & Human-Escalation Tradeoff).
   - `outputs/tables/` — CSVs for Experiment Log, Robustness Accuracy & Macro-F1 Matrices, Severity Sweep, Per-Class F1 Report, Top Confused Intent Pairs, Detailed Failure Cases, and Hand-Crafted Tested Queries.
   - `outputs/models/` — Saved `distilbert_clean`, `distilbert_p1_partial_aug`, and `distilbert_robust_final` (with `calibration.json` temperature parameter).
   - `outputs/reports/executive_summary.json` — All headline numbers in one JSON file for your PPT slides.

---

## Presenting Live to Your Professors

1. **Slide/Walkthrough Flow (Maps Directly to Handbook M1–M6):**
   - **M1 (Business Case):** 27-intent customer support routing + why clean accuracy hides real-world chat noise failures.
   - **M2 (Data & Privacy):** Bitext 27 intents, deduplication, leak-free 70/15/15 split, PII placeholders (`{{Order Number}}`) preserved.
   - **M3 (Models Compared):** TF-IDF + Logistic Regression, TF-IDF + Calibrated Linear SVM, 2-Layer BiLSTM + Self-Attention, DistilBERT (Clean), DistilBERT (Phase 1 2-Noise Aug), and DistilBERT (Phase 2 Full 5-Noise Robust + Label Smoothing + Temperature Calibration).
   - **M4 (Evaluation & Failure Analysis):** Show the Robustness Heatmap, Ablation bar chart (Phase 1 → Phase 2), 27×27 Confusion Matrices, Top Confused Sibling Intents, and Categorized Failure Modes.
   - **M5 (Live Demo — Section 12):** Scroll to the **Interactive Dashboard widget**, type a messy customer message (or ask a professor to suggest one!), toggle noise injection + escalation threshold `τ`, and show live predictions, confidence bars, latency, and auto-route vs. human-escalation decisions.
   - **M6 (Executive Summary):** Use `outputs/plots/*.png` and `outputs/reports/executive_summary.json` directly in your final PPT.
