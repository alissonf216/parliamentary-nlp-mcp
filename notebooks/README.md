# Notebooks & Experimental Pipeline

Reproducible **training / evaluation** suite for the parliamentary offensive-speech study. This is the quality layer recruiters and reviewers can inspect: protocol, code, tables, and figures.

| File | Role |
| --- | --- |
| [`finetune_bertimbau_huggingface.ipynb`](finetune_bertimbau_huggingface.ipynb) | **Train + save** BERTimbau (Flat-4) in Hugging Face format + optional Hub upload |
| [`experiments_hierarchy_imbalance.ipynb`](experiments_hierarchy_imbalance.ipynb) | Full comparative suite (all models, hierarchy, imbalance) |
| [`experimentos_pipeline.py`](experimentos_pipeline.py) | Engine behind the hierarchy / imbalance notebook |

Published metrics and plots from the completed comparative run:

- [`../docs/MODELING.md`](../docs/MODELING.md) — narrative + key charts
- [`../docs/results/`](../docs/results/) — CSV / JSON metrics
- [`../docs/figures/`](../docs/figures/) — heatmaps, bars, confusion matrices, ROC/PR

The MCP package under `src/` **serves** a classifier; it does not re-run this training loop. After fine-tuning with the notebook above, point the server at your Hub repo (or local folder):

```bash
export PARLIAMENTARY_NLP_MODEL_ID="alissonf216/parliamentary-bertimbau-auditor"
```

---

## Quick start — production fine-tune (BERTimbau only)

```bash
cd notebooks
jupyter notebook finetune_bertimbau_huggingface.ipynb
```

On Colab: upload the notebook, enable GPU, run all cells. Output lands in `outputs/parliamentary-bertimbau-auditor/` ready for Hub upload.

## Quick start — full experiment suite

```bash
cd notebooks
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch transformers scikit-learn pandas seaborn matplotlib sentencepiece openpyxl
jupyter notebook experiments_hierarchy_imbalance.ipynb
```

For a smoke test of the suite, set in the config cell:

```python
exp.QUICK_MODE = True
```
