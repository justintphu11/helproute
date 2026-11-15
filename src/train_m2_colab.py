"""Google Colab GPU trainer for HelpRoute M2. Never use the locked test set here."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

DATA = Path("train_labels.parquet")
OUT = Path("m2_distilroberta")


def compute_metrics(pred):
    logits, labels = pred
    p = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    return {"pr_auc": average_precision_score(labels, p)}


def main():
    df = pd.read_parquet(DATA)
    y = df["is_need"].astype(int).to_numpy()
    tr, va = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
                  .split(df, y, df["dup_key"].fillna(df["post_id"])))
    tok = AutoTokenizer.from_pretrained("distilroberta-base")

    def encode(rows):
        return tok(rows["text"], truncation=True, max_length=256)

    train = Dataset.from_pandas(df.iloc[tr][["text", "is_need"]].rename(columns={"is_need": "labels"}), preserve_index=False).map(encode, batched=True)
    valid = Dataset.from_pandas(df.iloc[va][["text", "is_need"]].rename(columns={"is_need": "labels"}), preserve_index=False).map(encode, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
    args = TrainingArguments(output_dir="m2_checkpoints", learning_rate=2e-5, per_device_train_batch_size=16,
        per_device_eval_batch_size=32, num_train_epochs=5, weight_decay=0.01, eval_strategy="epoch",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="pr_auc", seed=42, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()
    probs = torch.softmax(torch.tensor(trainer.predict(valid).predictions), dim=1)[:, 1].numpy()
    choices = np.unique(np.r_[0, probs, 1]); threshold = max(t for t in choices if (probs[y[va] == 1] >= t).mean() >= .90)
    precision, recall, f1, _ = precision_recall_fscore_support(y[va], probs >= threshold, average="binary", zero_division=0)
    trainer.save_model(OUT); tok.save_pretrained(OUT)
    Path("m2_threshold.json").write_text(json.dumps({"threshold": float(threshold)}, indent=2))
    Path("m2_val.md").write_text(f"# M2 validation\n\nPrecision {precision:.3f}; recall {recall:.3f}; F1 {f1:.3f}; PR-AUC {average_precision_score(y[va], probs):.3f}; threshold {threshold:.4f}.\n")


if __name__ == "__main__": main()
