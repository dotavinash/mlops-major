from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib

def main():
    print("[test] Loading Olivetti faces…")
    data = fetch_olivetti_faces()
    X = data.images.reshape(len(data.images), -1)
    y = data.target

    print("[test] Split 70/30 (stratified, rs=42) — MUST match train.py")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    model_path = Path("models") / "savedmodel.pth"
    assert model_path.exists(), f"Model not found: {model_path}"
    print(f"[test] Loading model from {model_path} …")
    clf = joblib.load(model_path)

    print("[test] Evaluating…")
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[test] Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
