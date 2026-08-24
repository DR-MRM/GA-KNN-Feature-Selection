"""GA-KNN Feature Selection — English GitHub version.

This file contains the final analysis workflow corresponding to Cells 17–22
of the validated study. Cells 14 and 15 are intentionally omitted because
they were test-only cells.

Required in-memory inputs before running the complete workflow:
    df_train, df_test

The data must contain the binary target column ``status`` and the original
22 predictor features. The GA-10 and Stable-7 feature sets below are the
validated sets reported by the experiment.
"""

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel, wilcoxon
from sklearn.metrics import (accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

TARGET_COL = "status"
N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
N_SPLITS = 5
N_REPEATS = 10
CV_RANDOM_STATE = 42
ROBUSTNESS_RANDOM_STATE = 2026
CORR_THRESHOLD = 0.80

GA10_FEATURES = ["MDVP:Fo(Hz)", "MDVP:Flo(Hz)", "MDVP:PPQ", "MDVP:Shimmer(dB)",
                 "Shimmer:APQ5", "NHR", "RPDE", "DFA", "D2", "PPE"]

STABLE7_FEATURES = ["D2", "spread1", "PPE", "MDVP:Shimmer(dB)", "HNR",
                    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"]


def build_knn():
    """Create the KNN classifier used throughout the experiment."""
    return KNeighborsClassifier(n_neighbors=N_NEIGHBORS, metric="euclidean",
                                 weights=KNN_WEIGHTS)


def get_oof_threshold(X, y):
    """Select a threshold from training OOF predictions only using Youden's J."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=CV_RANDOM_STATE)
    scores = cross_val_predict(build_knn(), X, y, cv=cv,
                               method="predict_proba")[:, 1]
    fpr, tpr, thresholds = roc_curve(y, scores)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        raise RuntimeError("No valid OOF threshold was found.")
    idx = np.where(valid)[0]
    best = idx[np.argmax((tpr - fpr)[idx])]
    return (float(thresholds[best]), average_precision_score(y, scores),
            roc_auc_score(y, scores))


def evaluate_independent_test(X_train, X_test, y_train, y_test):
    """Fit on train data and evaluate once on the independent test set."""
    threshold, oof_pr, oof_roc = get_oof_threshold(X_train, y_train)
    model = build_knn().fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold, "OOF_PR_AUC": oof_pr, "OOF_ROC_AUC": oof_roc,
        "PR_AUC": average_precision_score(y_test, score),
        "ROC_AUC": roc_auc_score(y_test, score),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "Specificity": tn / max(tn + fp, 1),
        "F1": f1_score(y_test, pred, zero_division=0),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "Confusion_Matrix": cm,
    }


def repeated_cv(X, y, model_name, cv):
    """Run repeated stratified CV with a fixed 0.5 decision threshold."""
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        model = build_knn().fit(X.iloc[tr], y[tr])
        score = model.predict_proba(X.iloc[va])[:, 1]
        pred = (score >= 0.5).astype(int)
        rows.append({"model": model_name, "fold": fold,
                     "PR_AUC": average_precision_score(y[va], score),
                     "ROC_AUC": roc_auc_score(y[va], score),
                     "Accuracy": accuracy_score(y[va], pred),
                     "Precision": precision_score(y[va], pred, zero_division=0),
                     "Recall": recall_score(y[va], pred, zero_division=0),
                     "F1": f1_score(y[va], pred, zero_division=0)})
    return pd.DataFrame(rows)


def summarize_cv(results_cv):
    metrics = ["PR_AUC", "ROC_AUC", "Accuracy", "Precision", "Recall", "F1"]
    rows = []
    for model in ["Baseline-22", "GA-10"]:
        d = results_cv[results_cv.model == model]
        row = {"Model": model}
        for m in metrics:
            row[f"{m}_Mean"] = d[m].mean()
            row[f"{m}_Std"] = d[m].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_statistics(results_cv):
    """Paired Wilcoxon, paired t-test, 95% CI and Cohen's dz."""
    metrics = ["PR_AUC", "ROC_AUC", "Accuracy", "Precision", "Recall", "F1"]
    b = results_cv[results_cv.model == "Baseline-22"].sort_values("fold").reset_index(drop=True)
    g = results_cv[results_cv.model == "GA-10"].sort_values("fold").reset_index(drop=True)
    rows = []
    for m in metrics:
        x, y = b[m].to_numpy(float), g[m].to_numpy(float)
        d = y - x
        mean_d, med_d, sd = d.mean(), np.median(d), d.std(ddof=1)
        se = sd / np.sqrt(len(d))
        tr = ttest_rel(y, x)
        crit = t.ppf(.975, len(d)-1)
        try:
            wr = wilcoxon(y, x, alternative="two-sided", zero_method="wilcox", method="auto")
            wp, ws = float(wr.pvalue), float(wr.statistic)
        except Exception:
            wp, ws = np.nan, np.nan
        rows.append({"Metric": m, "Mean_Difference": mean_d,
                     "Median_Difference": med_d,
                     "CI95_Low": mean_d-crit*se, "CI95_High": mean_d+crit*se,
                     "Wilcoxon_Statistic": ws, "Wilcoxon_p": wp,
                     "Paired_t_p": float(tr.pvalue),
                     "Cohens_dz": mean_d/sd if sd > 1e-12 else 0.0,
                     "Direction": "GA-10 better" if mean_d > 0 else "Baseline better" if mean_d < 0 else "Equal",
                     "Significance": "Significant" if wp < .05 else "Not significant"})
    return pd.DataFrame(rows)


def feature_overlap(baseline_features, ga_features=GA10_FEATURES, stable_features=STABLE7_FEATURES):
    """Calculate GA/Stable overlap and dimensionality reduction."""
    ga, st, base = set(ga_features), set(stable_features), set(baseline_features)
    overlap = sorted(ga & st)
    return {"overlap": overlap, "ga_only": sorted(ga-st), "stable_only": sorted(st-ga),
            "jaccard": len(ga&st)/len(ga|st),
            "ga_reduction_pct": (1-len(ga)/len(base))*100,
            "stable_reduction_pct": (1-len(st)/len(base))*100}


def correlation_analysis(df_train, baseline_features, ga_features=GA10_FEATURES,
                          threshold=CORR_THRESHOLD):
    """Compute Pearson redundancy statistics using training data only."""
    X = df_train[list(baseline_features)].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all").fillna(X.median())
    corr = X.corr()

    def stats(m):
        a = m.to_numpy(float); mask = np.triu(np.ones(a.shape, bool), 1)
        v = np.abs(a[mask]); v = v[np.isfinite(v)]
        return (float(v.mean()), float(v.max()), int((v >= threshold).sum())) if len(v) else (np.nan,np.nan,0)

    base_stats = stats(corr)
    available = [f for f in ga_features if f in corr.columns]
    ga_corr = corr.loc[available, available]
    ga_stats = stats(ga_corr)
    return {"corr_matrix": corr, "ga_corr_matrix": ga_corr,
            "baseline_mean_abs_corr": base_stats[0], "baseline_max_abs_corr": base_stats[1],
            "baseline_high_corr_count": base_stats[2], "ga_mean_abs_corr": ga_stats[0],
            "ga_max_abs_corr": ga_stats[1], "ga_high_corr_count": ga_stats[2]}


def run_analysis(df_train, df_test, target_col=TARGET_COL):
    """Run independent testing, robustness CV, statistics, overlap and redundancy."""
    features = [c for c in df_train.columns if c != target_col]
    missing = [f for f in GA10_FEATURES + STABLE7_FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing validated features: {missing}")
    y_train = df_train[target_col].astype(int).to_numpy()
    y_test = df_test[target_col].astype(int).to_numpy()
    feature_sets = {"Baseline KNN": features, "GA-KNN": GA10_FEATURES, "Stable-KNN": STABLE7_FEATURES}

    rows = []
    for name, fs in feature_sets.items():
        r = evaluate_independent_test(df_train[fs], df_test[fs], y_train, y_test)
        rows.append({"Model": name, "Features": len(fs), **{k:r[k] for k in
                     ["PR_AUC","ROC_AUC","Accuracy","Precision","Recall","Specificity","F1"]}})
    independent = pd.DataFrame(rows)

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=2026)
    results_cv = pd.concat([
        repeated_cv(df_train[features], y_train, "Baseline-22", cv),
        repeated_cv(df_train[GA10_FEATURES], y_train, "GA-10", cv)], ignore_index=True)

    return {"independent_test": independent,
            "results_cv": results_cv,
            "robustness_summary": summarize_cv(results_cv),
            "statistics": paired_statistics(results_cv),
            "feature_overlap": feature_overlap(features),
            "correlation": correlation_analysis(df_train, features)}


# ---------------------------------------------------------------------------
# Validated numerical results from the final experiment
# ---------------------------------------------------------------------------
FINAL_INDEPENDENT_RESULTS = pd.DataFrame([
    ["Baseline KNN",22,.9989,.9966,.8974,1.0,.8621,1.0,.9259],
    ["GA-KNN",10,.9949,.9828,.8718,1.0,.8276,1.0,.9057],
    ["Stable-KNN",7,.9704,.9138,.7179,1.0,.6207,1.0,.7660],
], columns=["Model","Features","PR-AUC","ROC-AUC","Accuracy","Precision","Recall","Specificity","F1"])

FINAL_ROBUSTNESS = pd.DataFrame([
    ["Baseline-22","0.9869 ± 0.0224","0.9661 ± 0.0454","0.9181 ± 0.0513","0.9681 ± 0.0393","0.9473 ± 0.0325"],
    ["GA-10","0.9866 ± 0.0191","0.9636 ± 0.0420","0.9013 ± 0.0547","0.9478 ± 0.0452","0.9359 ± 0.0350"],
], columns=["Model","PR-AUC","ROC-AUC","Accuracy","Recall","F1"])

FINAL_STATISTICS = pd.DataFrame([
    ["PR-AUC",-.000234,.153953,"Not significant"],
    ["ROC-AUC",-.002546,.177125,"Not significant"],
    ["Accuracy",-.016774,.033476,"Significant"],
    ["Precision",-.002379,.455200,"Not significant"],
    ["Recall",-.020290,.001620,"Significant"],
    ["F1",-.011386,.017711,"Significant"],
], columns=["Metric","GA_minus_Baseline","Wilcoxon_p","Result"])


def print_final_summary():
    """Print the publication-oriented final decision."""
    print("="*90)
    print("GA-KNN FEATURE SELECTION — FINAL SCIENTIFIC SUMMARY")
    print("="*90)
    print("Dataset: 195 samples | Train: 156 | Independent test: 39")
    print("Original features: 22 | GA-10: 10 | Stable-7: 7")
    print("KNN: k=5, distance weighting")
    print("GA: population=30, generations=20, roulette-wheel selection, two-point crossover, Gaussian mutation, elitism=2")
    print("\nIndependent test performance:\n")
    print(FINAL_INDEPENDENT_RESULTS.to_string(index=False))
    print("\nRepeated-CV robustness:\n")
    print(FINAL_ROBUSTNESS.to_string(index=False))
    print("\nPaired statistical comparison:\n")
    print(FINAL_STATISTICS.to_string(index=False))
    print("\nFinal decision:")
    print("GA reduced dimensionality and feature redundancy but did not improve predictive performance.")
    print("Baseline-22 is the conservative predictive model; GA-10 is the preferred feature-reduced alternative.")


if __name__ == "__main__":
    print_final_summary()
