"""Train and evaluate both models. Saves the best model, metrics and graphs.

Run:  python train.py
"""
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import db
import pipeline as pl

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def main():
    df = pl.load_data()
    X, y = pl.split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=pl.RANDOM_STATE)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}  Attrition rate: {y.mean():.3f}")

    models = pl.build_models(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=pl.RANDOM_STATE)
    results, probs = {}, {}
    db.init_db()

    for name, model in models.items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, p),
            "cv_roc_auc_mean": float(cv_auc.mean()),
            "cv_roc_auc_std": float(cv_auc.std()),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        probs[name] = p
        db.save_model_info(name, results[name]["accuracy"], results[name]["f1"],
                           results[name]["roc_auc"])
        joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.pkl")
        r = results[name]
        print(f"{name}: acc={r['accuracy']:.3f} prec={r['precision']:.3f} rec={r['recall']:.3f} "
              f"f1={r['f1']:.3f} auc={r['roc_auc']:.3f} cvAUC={r['cv_roc_auc_mean']:.3f}")

    # baseline: always predict "stays"
    results["Baseline (always No)"] = {"accuracy": float(1 - y_te.mean()), "recall": 0.0}

    best = max(["Logistic Regression", "Random Forest"], key=lambda n: results[n]["f1"])
    joblib.dump(models[best], "models/best_model.pkl")
    results["best_model"] = best
    print("Best model (by F1):", best)

    # feature importance (Random Forest) and coefficients (Logistic Regression)
    rf = models["Random Forest"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = rf.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:10]
    results["top_features_rf"] = [(str(names[i]).split("__")[1], float(imp[i])) for i in order]
    lr = models["Logistic Regression"]
    coef = lr.named_steps["clf"].coef_[0]
    lnames = lr.named_steps["prep"].get_feature_names_out()
    lorder = np.argsort(np.abs(coef))[::-1][:10]
    results["top_features_lr"] = [(str(lnames[i]).split("__")[1], float(coef[i])) for i in lorder]

    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # graphs (black and white friendly)
    plt.figure(figsize=(5, 4))
    for (name, p), ls in zip(probs.items(), ["-", "--"]):
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, ls, color="black", label=f"{name} (AUC={results[name]['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], ":", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, name in zip(axes, ["Logistic Regression", "Random Forest"]):
        cm = np.array(results[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Greys")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"]); ax.set_yticklabels(["No", "Yes"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=9)
    plt.tight_layout(); plt.savefig("results/confusion_matrices.png", dpi=150); plt.close()

    plt.figure(figsize=(5.5, 3.6))
    feats = [f for f, _ in results["top_features_rf"]][::-1]
    vals = [v for _, v in results["top_features_rf"]][::-1]
    plt.barh(feats, vals, color="gray", edgecolor="black")
    plt.xlabel("Importance"); plt.title("Top 10 features (Random Forest)")
    plt.tight_layout(); plt.savefig("results/feature_importance.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
