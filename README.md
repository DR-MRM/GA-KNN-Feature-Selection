# GA-KNN Feature Selection

## Genetic Algorithm-Based Feature Selection with K-Nearest Neighbors for Parkinson's Disease Classification

This repository presents an experimental framework for **genetic algorithm (GA)-based feature selection combined with a K-Nearest Neighbors (KNN) classifier** for Parkinson's disease classification.

The primary objective is to investigate whether a genetic algorithm can reduce the dimensionality and redundancy of the original feature space while preserving predictive performance.

Rather than assuming that feature selection necessarily improves classification accuracy, this study evaluates the **trade-off between dimensionality reduction, feature redundancy, robustness, and predictive performance** using independent testing and repeated cross-validation.

---

## Research Objective

High-dimensional feature spaces may contain redundant or weakly informative variables. Feature selection can potentially:

- Reduce the number of input features
- Remove redundant information
- Simplify the classification model
- Improve interpretability
- Reduce computational complexity
- Preserve the most discriminative information

In this study, a **Genetic Algorithm (GA)** is used to search for informative feature subsets, while **KNN** is used as the downstream classifier.

The central research question is:

> **Can GA-based feature selection substantially reduce the feature space while maintaining comparable predictive performance to the full-feature KNN baseline?**

---

## Experimental Design

The dataset contains:

| Property | Value |
|---|---:|
| Total samples | 195 |
| Training samples | 156 |
| Independent test samples | 39 |
| Original features | 22 |
| Classifier | KNN |
| Number of neighbors | 5 |
| Weighting | Distance |
| Cross-validation | Stratified 5-Fold |
| GA population | 30 |
| GA generations | 20 |
| Selection | Roulette Wheel |
| Crossover | Two-Point |
| Mutation | Gaussian |
| Elitism | 2 |

The dataset is divided into a training set and an **independent test set**. The independent test set is not used during feature selection, robustness validation, or threshold optimization.

---

## Methodology

The experimental workflow consists of the following stages:

```text
Original Dataset
       │
       ▼
Data Cleaning & Preprocessing
       │
       ▼
Stratified Train/Test Split
       │
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
Baseline KNN                    Genetic Algorithm
22 Features                    Feature Selection
       │                              │
       │                              ▼
       │                         GA-Selected
       │                         Feature Subset
       │                              │
       └──────────────┬───────────────┘
                      ▼
                 KNN Evaluation
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
 Independent Test          Repeated CV
 Evaluation                Robustness Analysis
          │                       │
          └───────────┬───────────┘
                      ▼
             Statistical Comparison
                      │
                      ▼
             Scientific Interpretation
```

---

## Genetic Algorithm Feature Selection

Each GA chromosome represents a candidate feature subset.

A binary representation is used:

```text
1 → feature selected
0 → feature not selected
```

The GA searches the feature space using:

- Roulette-wheel selection
- Two-point crossover
- Gaussian mutation
- Elitism
- KNN-based fitness evaluation

The selected subset is subsequently evaluated independently against the original 22-feature baseline.

---

## Selected Feature Subsets

### GA-10

The final GA-selected subset contains 10 features:

1. `MDVP:Fo(Hz)`
2. `MDVP:Flo(Hz)`
3. `MDVP:PPQ`
4. `MDVP:Shimmer(dB)`
5. `Shimmer:APQ5`
6. `NHR`
7. `RPDE`
8. `DFA`
9. `D2`
10. `PPE`

This represents a reduction from 22 to 10 features:

**54.5% dimensionality reduction**

---

### Stable-7

Repeated GA runs were also used to investigate feature-selection stability.

The stable subset contains seven frequently selected features:

1. `D2`
2. `spread1`
3. `PPE`
4. `MDVP:Shimmer(dB)`
5. `HNR`
6. `MDVP:Fo(Hz)`
7. `MDVP:Fhi(Hz)`

This represents a:

**68.2% dimensionality reduction**

The Stable-7 subset is evaluated as an additional feature-reduced configuration rather than as the primary GA result.

---

## Evaluation Strategy

The study uses several complementary evaluation strategies.

### 1. Independent Test Evaluation

The following models are compared:

- Baseline KNN — 22 features
- GA-KNN — 10 features
- Stable-KNN — 7 features

The independent test set is kept separate from the model-selection process.

Evaluation metrics include:

- PR-AUC
- ROC-AUC
- Accuracy
- Precision
- Recall
- Specificity
- F1-score

---

### 2. Repeated Cross-Validation

To evaluate robustness, Baseline-22 and GA-10 are compared using:

**10 × Repeated Stratified 5-Fold Cross-Validation**

This produces 50 paired validation results for each model.

The following metrics are analyzed:

- PR-AUC
- ROC-AUC
- Accuracy
- Precision
- Recall
- F1-score

Results are reported as:

```text
mean ± standard deviation
```

---

### 3. Paired Statistical Comparison

Because the two models are evaluated on the same cross-validation splits, paired statistical comparisons are performed.

The primary statistical test is:

**Wilcoxon signed-rank test**

A paired t-test is also calculated as a complementary analysis.

Effect size is reported using:

**Paired Cohen's dₓ**

The primary performance metric is:

**PR-AUC**

This metric is emphasized because it provides a useful threshold-independent assessment of classification performance, particularly when class distributions are not perfectly balanced.

---

## Main Results

### Independent Test Set

| Model | Features | PR-AUC | ROC-AUC | Accuracy | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline KNN | 22 | 0.9989 | 0.9966 | 0.8974 | 1.0000 | 0.8621 | 1.0000 | 0.9259 |
| GA-KNN | 10 | 0.9949 | 0.9828 | 0.8718 | 1.0000 | 0.8276 | 1.0000 | 0.9057 |
| Stable-KNN | 7 | 0.9704 | 0.9138 | 0.7179 | 1.0000 | 0.6207 | 1.0000 | 0.7660 |

The full 22-feature baseline achieves the strongest predictive performance on the independent test set.

GA-KNN, however, achieves very high predictive performance while using less than half of the original features.

---

## Repeated-CV Robustness

| Model | PR-AUC | ROC-AUC | Accuracy | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline-22 | 0.9869 ± 0.0224 | 0.9661 ± 0.0454 | 0.9181 ± 0.0513 | 0.9681 ± 0.0393 | 0.9473 ± 0.0325 |
| GA-10 | 0.9866 ± 0.0191 | 0.9636 ± 0.0420 | 0.9013 ± 0.0547 | 0.9478 ± 0.0452 | 0.9359 ± 0.0350 |

The difference in PR-AUC is very small:

```text
Baseline-22 : 0.9869
GA-10       : 0.9866
Difference  : -0.0002
```

The Wilcoxon test gives:

```text
p = 0.1540
```

Therefore, the difference in PR-AUC is **not statistically significant** at the 0.05 level.

---

## Statistical Comparison

| Metric | GA − Baseline | Wilcoxon p-value | Result |
|---|---:|---:|---|
| PR-AUC | -0.000234 | 0.153953 | Not significant |
| ROC-AUC | -0.002546 | 0.177125 | Not significant |
| Accuracy | -0.016774 | 0.033476 | Significant |
| Precision | -0.002379 | 0.455200 | Not significant |
| Recall | -0.020290 | 0.001620 | Significant |
| F1 | -0.011386 | 0.017711 | Significant |

These results indicate that GA-based feature selection **does not provide a statistically significant predictive-performance improvement** in this experiment.

Instead, its main benefit is dimensionality reduction.

---

## Feature Stability

Ten independent GA runs were used to investigate the stability of feature selection.

A feature was considered stable when it appeared in at least **40% of the independent GA runs**.

Seven stable features were identified.

The most frequently selected feature was:

```text
D2 — 50%
```

The Stable-7 subset provides evidence that several features are repeatedly favored by the evolutionary search, although the reduced Stable-7 model produces lower predictive performance than both the full baseline and GA-10 models.

---

## Feature Overlap

The GA-10 and Stable-7 subsets share several features, indicating partial consistency between the evolutionary search and the stability analysis.

The overlapping features are:

- `D2`
- `PPE`
- `MDVP:Shimmer(dB)`
- `MDVP:Fo(Hz)`

Thus:

```text
GA-10 / Stable-7 overlap = 4 features
GA-10 overlap            = 40.0%
Stable-7 overlap         = 57.1%
Jaccard similarity       ≈ 0.308
```

This overlap provides additional evidence that some variables are repeatedly identified as useful by the feature-selection process.

---

## Feature Redundancy Analysis

Pearson correlation analysis was performed using **training data only**.

A feature pair was considered highly correlated when:

```text
|r| ≥ 0.80
```

The observed redundancy was:

| Configuration | Mean |r| | High-Correlation Pairs |
|---|---:|---:|
| Baseline-22 | 0.5043 | 40 |
| GA-10 | 0.3737 | 3 |

Therefore, GA selection produced:

```text
High-correlation pair reduction : 92.5%
Mean absolute correlation reduction : 25.9%
```

This is an important finding because it demonstrates that the GA-selected subset is substantially less redundant than the original feature space.

---

## Scientific Interpretation

The results support the following conclusions:

1. The genetic algorithm reduced the feature space from **22 to 10 features**.
2. This corresponds to a **54.5% reduction in dimensionality**.
3. GA-10 preserved almost the same PR-AUC as the full-feature baseline under repeated cross-validation.
4. The difference in PR-AUC was not statistically significant.
5. ROC-AUC also showed no statistically significant difference.
6. Accuracy, Recall, and F1-score were significantly lower for GA-10.
7. Therefore, the GA did **not demonstrate predictive performance improvement** in this experiment.
8. The GA substantially reduced feature redundancy.
9. The GA-selected subset should therefore primarily be interpreted as a **feature-reduced representation**, rather than a performance-enhancement mechanism.
10. Baseline KNN remains the preferred configuration when maximum predictive performance is the primary objective.
11. GA-KNN provides a useful alternative when dimensionality reduction, lower redundancy, and model simplification are important objectives.

---

## Key Finding

The central finding of this experiment can be summarized as:

> **GA-based feature selection successfully reduced the feature space and substantially decreased feature redundancy while preserving statistically comparable PR-AUC, but it did not improve predictive performance.**

This distinction is important for interpreting the contribution of evolutionary feature selection. A successful feature-selection method does not necessarily need to increase predictive accuracy; it may instead provide a more compact representation with comparable predictive information.

---

## Reference Study

The experimental design is conceptually related to the following work:

> Pei, M., Goodman, E. D., & Punch, W. F. (1998). *Feature Extraction Using Genetic Algorithms.*

The present experiment follows the general concept of coupling **genetic feature search with KNN classification**, while extending the evaluation through independent testing, repeated cross-validation, feature stability analysis, statistical comparison, and correlation-based redundancy analysis.

---

## Repository Structure

```text
GA-KNN-Feature-Selection/
│
├── ga_knn_feature_selection.py
├── README.md
└── requirements.txt
```

The repository intentionally provides the experimental Python implementation separately from the original research notebook.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/DR-MRM/GA-KNN-Feature-Selection.git
cd GA-KNN-Feature-Selection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Technologies

The implementation uses Python and common scientific machine-learning libraries, including:

- Python
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

---

## Reproducibility

The experiment uses fixed random seeds for the principal validation procedures where applicable.

The independent test set is kept separate from the feature-selection and robustness-validation procedures.

Correlation and redundancy analysis is performed exclusively on the training data to avoid information leakage from the independent test set.

---

## Limitations

Several limitations should be considered when interpreting these results:

- The dataset contains only 195 observations.
- KNN performance can be sensitive to feature scaling and distance structure.
- Feature selection results from stochastic genetic algorithms may vary across runs.
- Statistical comparisons are based on repeated cross-validation folds rather than independent datasets.
- The observed results should not be interpreted as evidence that GA-based feature selection universally improves Parkinson's disease classification.
- External validation on an independent dataset is required before drawing conclusions about generalization.

---

## Research Status

**Status: Completed experimental study**

The current repository represents the completed GA-KNN feature-selection experiment, including:

- Data preprocessing
- Stratified train/test splitting
- GA feature selection
- Independent test evaluation
- Feature-count comparison
- Feature stability analysis
- Baseline vs. GA comparison
- Repeated cross-validation
- Paired statistical testing
- Feature-overlap analysis
- Correlation/redundancy analysis
- Final scientific interpretation

---

## Citation

If you use this repository or build upon this experimental framework, please cite the repository and the associated research work when available.

```bibtex
@software{ga_knn_feature_selection,
  author = {DR-MRM},
  title = {GA-KNN Feature Selection},
  year = {2026},
  url = {https://github.com/DR-MRM/GA-KNN-Feature-Selection}
}
```

---

## License

A license will be added to the repository according to the intended research and code-distribution policy.