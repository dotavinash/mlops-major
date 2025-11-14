from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib

def main():
    print("[train] Loading Olivetti faces…")
    data = fetch_olivetti_faces()
    X = data.images.reshape(len(data.images), -1)
    y = data.target

    print("[train] Split 70/30 (stratified, rs=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    print("[train] Training DecisionTreeClassifier…")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    print("[train] Evaluating on test set…")
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[train] Test accuracy (one-shot check): {acc:.4f}")

    Path("models").mkdir(exist_ok=True)
    out_path = Path("models") / "savedmodel.pth"
    joblib.dump(clf, out_path)
    print(f"[train] Model saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
