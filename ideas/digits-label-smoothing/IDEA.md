# IDEA: label smoothing for a small MLP on sklearn digits

Hypothesis: label smoothing and small architecture/regularization changes improve the test accuracy of a 2-layer
MLP trained for a fixed 1,500 optimizer steps (batch 64) on `sklearn.datasets.load_digits` (CPU only, a few
seconds per run). Metric: accuracy, maximize; meaningful effect: 0.05 (5 points). Data: split the 1,797 digits
50/25/25 into train / validation / holdout with a fixed seed. Editable: the model, optimizer and training step.
Budget: exactly 1,500 steps, run by the frozen harness.

---
Orchestrator notes (pack authoring, docs/PACK-AUTHORING.md): item pack (one item per digit image), 898 train /
449 validation / 450 holdout. Expected holdout SE ≈ √(0.2/450 + 2σ²/3) ≈ 0.02 for σ ≈ 0.005 → 2·SE ≈ 0.04 < MES 0.05
(powered, though below the recommended 2.5·SE margin; MES is the owner's). Campaign ≤ 30 min (max_hours 0.5).
