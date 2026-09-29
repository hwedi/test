---
title: Transformer Health Check
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Transformer Health Check

Assesses a power transformer from its dissolved gas readings (H2, CO, C2H4, C2H2):

- **Fault diagnosis (FDD):** predicted condition, confidence, and a manual-review flag when confidence is low
- **Remaining useful life (RUL):** estimated days left, with a ±range

Input: a CSV with 420 rows (one reading every 12 hours) and columns H2, CO, C2H4, C2H2.

Folders:
- `artifacts/` trained models and preprocessing exported from the Colab notebook
- `samples/` example test transformers and their true labels, used by the demo tab

Decision-support tool built for a student capstone project. It does not replace engineering judgement.
