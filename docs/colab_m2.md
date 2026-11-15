# M2 on Google Colab

Use a GPU runtime. Upload only `data/processed/train_labels.parquet` and `src/train_m2_colab.py`; do not upload `labeling/test_final.csv`.

```python
!pip install -q "transformers>=4.46,<5" datasets accelerate pyarrow scikit-learn
!python train_m2_colab.py
```

Download `m2_distilroberta/`, `m2_threshold.json`, and `m2_val.md` after training. Place them in the local project as `models/m2_distilroberta/`, `models/m2_threshold.json`, and `reports/m2_val.md`.
