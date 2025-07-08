# train_classifier.py  (Random‑Forest)
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# —— load dataset ——————————————————————————————————————————
with open("data.pickle", "rb") as f:
    d = pickle.load(f)

raw_X = d["data"]
raw_y = d["labels"]
classes = np.asarray(d["label_encoder"])

# —— filter only samples with length 42 ——————————————————————
EXPECTED_LEN = 42
filtered = [(x, y) for x, y in zip(raw_X, raw_y) if len(x) == EXPECTED_LEN]

if len(filtered) == 0:
    raise ValueError("[ERROR] No valid samples with 42 features found.")

X = np.asarray([x for x, _ in filtered], dtype=np.float32)
y = np.asarray([y for _, y in filtered], dtype=np.int32)

print(f"[INFO] Loaded {len(X)} clean samples.")

# —— split ————————————————————————————————————————————————
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# —— model ————————————————————————————————————————————————
rf = RandomForestClassifier(
    n_estimators=400,
    n_jobs=-1,
    random_state=42
)
rf.fit(Xtr, ytr)

# —— evaluation ————————————————————————————————————————————
acc = accuracy_score(yte, rf.predict(Xte))
print(f"{acc*100:.2f}% accuracy (Random‑Forest)")

# —— save ————————————————————————————————————————————————
with open("model.p", "wb") as f:
    pickle.dump({"model": rf, "classes": classes}, f)

print("[INFO] model.p written")
