import numpy as np
import pickle
import yaml
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class InstrumentPredictor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.window      = cfg["prediction"]["window_size"]
        self.instruments = cfg["instruments"]
        self.le          = LabelEncoder().fit(self.instruments)
        self.model       = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained     = False
        self.cache       = Path("models/predictor.pkl")

    def _make_xy(self, sequence):
        valid = [l for l in sequence if l in self.instruments]
        enc   = self.le.transform(valid)
        X, y  = [], []
        for i in range(len(enc) - self.window):
            X.append(enc[i : i + self.window])
            y.append(enc[i + self.window])
        return np.array(X), np.array(y)

    def train(self, df):
        if df.empty or len(df) < self.window + 5:
            return
        seq = df.sort_values("frame_id")["label"].tolist()
        X, y = self._make_xy(seq)
        if len(X) < 5:
            return
        self.model.fit(X, y)
        self.trained = True
        Path("models").mkdir(exist_ok=True)
        with open(self.cache, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Predictor retrained.")

    def load(self):
        if self.cache.exists():
            with open(self.cache, "rb") as f:
                self.model = pickle.load(f)
            self.trained = True
            logger.info("Predictor loaded from cache.")

    def predict_next(self, recent):
        uniform = {i: round(1 / len(self.instruments), 3) for i in self.instruments}
        if not self.trained:
            return uniform
        valid = [l for l in recent if l in self.instruments]
        if len(valid) < self.window:
            valid = (self.instruments * (self.window + 1))[:self.window]
        enc   = self.le.transform(valid[-self.window:]).reshape(1, -1)
        proba = self.model.predict_proba(enc)[0]
        classes = self.le.inverse_transform(self.model.classes_)
        result  = {c: round(float(p), 3) for c, p in zip(classes, proba)}
        for i in self.instruments:
            result.setdefault(i, 0.0)
        return result

    def waste_probability(self, stats):
        return {
            inst: round(min(1.0, s["idle_ratio"] * 0.6 + (0.4 if s["is_waste"] else 0.0)), 3)
            for inst, s in stats.items()
        }