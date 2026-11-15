"""HelpRoute prediction API: the one thing the Streamlit app needs to import.

    from src.predict import HelpRoute
    hr = HelpRoute()                      # loads models once (cache this in Streamlit)
    hr.predict("I got evicted and have nowhere to sleep tonight")
    -> {"is_need": True, "need_prob": 0.97, "threshold": 0.037,
        "types": {"housing": 0.81, "homelessness": 0.64, ...},   # sorted, highest first
        "top_types": ["housing", "homelessness"],
        "evidence": ["evicted", "nowhere to sleep"],               # phrases that pushed toward "need"
        "crisis": False, "crisis_categories": [], "route": "resources"}

Models (from the Drive zip, unzipped into helproute/models/):
- is_need   : M2 + silver, fine-tuned DistilRoBERTa (models/m2_silver_distilroberta) - the tested final model
- types     : TF-IDF + logistic regression per type, trained on 2,033 silver + gold needs (models/types_silver.joblib;
              CV F1 in reports/types_cv.md). Falls back to the weaker M1 type models if missing.
- evidence  : top TF-IDF n-grams from M1's is_need model that appear in the text
- crisis    : src/safety.py keyword rule -> route to 988/911, not a resource reply
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from src.safety import detect
except ImportError:  # running from inside src/
    from safety import detect

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


class HelpRoute:
    def __init__(self, models_dir: Path = MODELS, device: str | None = None):
        m2_dir = models_dir / "m2_silver_distilroberta"
        if not m2_dir.exists():
            raise FileNotFoundError(f"{m2_dir} not found. Download helproute_models.zip from Drive and unzip it inside the helproute folder.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(m2_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(m2_dir).to(self.device).eval()
        self.threshold = json.loads((models_dir / "m2_silver_threshold.json").read_text())["threshold"]
        types_file = models_dir / "types_silver.joblib"
        self.type_models = joblib.load(types_file if types_file.exists() else models_dir / "m1_types.joblib")
        m1 = joblib.load(models_dir / "m1_is_need.joblib")
        self._tfidf, self._coef = m1.named_steps["tfidf"], m1.named_steps["clf"].coef_[0]
        self._vocab = np.array(self._tfidf.get_feature_names_out())

    def need_prob(self, texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self.tok(texts[i:i + 32], truncation=True, padding=True, max_length=256, return_tensors="pt").to(self.device)
                out.extend(torch.softmax(self.model(**enc).logits, 1)[:, 1].cpu().numpy())
        return np.array(out)

    def evidence(self, text: str, k: int = 5) -> list[str]:
        x = self._tfidf.transform([text])
        idx = x.indices[np.argsort(-(x.data * self._coef[x.indices]))]
        keep = lambda ng: any(w not in ENGLISH_STOP_WORDS and len(w) > 2 for w in ng.split())
        return [self._vocab[i] for i in idx if self._coef[i] > 0 and keep(self._vocab[i])][:k]

    def predict(self, text: str) -> dict:
        text = text or ""
        p = float(self.need_prob([text])[0])
        is_need = p >= self.threshold
        types = {name: float(m.predict_proba([text])[0, 1]) for name, m in self.type_models.items()} if is_need else {}
        types = dict(sorted(types.items(), key=lambda kv: -kv[1]))
        top = [t for t, v in types.items() if v >= 0.5] or list(types)[:1]
        safety = detect(text)
        return {
            "is_need": bool(is_need), "need_prob": round(p, 4), "threshold": round(self.threshold, 4),
            "types": {k: round(v, 3) for k, v in types.items()}, "top_types": top,
            "evidence": self.evidence(text) if is_need else [],
            "crisis": bool(safety["crisis"]), "crisis_categories": safety["categories"],
            "route": "crisis" if safety["crisis"] else ("resources" if is_need else "none"),
        }


if __name__ == "__main__":
    hr = HelpRoute()
    for t in ["I got evicted and have nowhere to sleep tonight, landlord changed the locks",
              "Laid off last month, unemployment ran out and I can't pay PG&E, shutoff notice came",
              "What's the best budgeting app for a family of four?",
              "I want to end it all, I can't go on like this"]:
        print(t[:60], "->", hr.predict(t))
