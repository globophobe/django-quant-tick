"""Random Forest Early Stopping - Find the Right Number of Trees Automatically

More trees usually means better performance, but only up to a point. After that, you're
wasting time training trees that don't help (and might even hurt if you overfit).

This module implements the fastai approach: train incrementally (50 trees, 100 trees,
150 trees, ...) and monitor cross-validation performance. When adding more trees stops
improving the CV score, stop training.

How it works:
- Start with min_estimators (e.g., 100 trees)
- Add step trees each iteration (e.g., +50 trees)
- Evaluate on cross-validation folds
- Track improvement: new_score - best_score
- If improvement < epsilon for `patience` consecutive iterations, stop
- Return the n_estimators that gave the best CV score

This saves training time and prevents overfitting from using too many trees. Inspired
by fastai's approach to training models incrementally and watching for convergence.
"""
import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def find_optimal_n_estimators(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    kfold: Any,
    max_estimators: int = 1000,
    min_estimators: int = 100,
    step: int = 50,
    epsilon: float = 0.001,
    patience: int = 2,
    max_features: str | int | float = "sqrt",
    min_samples_leaf: int = 50,
    max_depth: int | None = None,
    max_samples: float | None = None,
    random_state: int = 42,
) -> tuple[int, list[dict]]:
    """Automatically find the best number of trees by watching CV performance converge.

    Process: Train with min_estimators trees, evaluate CV AUC. Add `step` more trees,
    evaluate again. Keep going until adding more trees doesn't improve AUC by at least
    `epsilon` for `patience` consecutive iterations.

    This is much better than guessing a number like 500 or 1000. Maybe you only need
    200 trees (saves time), or maybe you need 800 trees (better performance). This
    function figures it out automatically.

    Supports both binary and multi-class classification. For multi-class (e.g., AFML's
    {-1, 0, +1} labels), uses one-vs-rest AUC with weighted average.

    Args:
        X: Feature matrix (samples × features)
        y: Target labels
        sample_weight: Sample weights for training
        kfold: Cross-validation splitter (PurgedKFold for time series)
        max_estimators: Don't go above this many trees (safety limit)
        min_estimators: Start with this many trees (don't stop before reaching this)
        step: How many trees to add each iteration
        epsilon: Improvement threshold (stop if gain < this for `patience` rounds)
        patience: How many consecutive no-improvement iterations before stopping
        max_features: RF parameter - features per split
        min_samples_leaf: RF parameter - minimum leaf size
        max_depth: RF parameter - tree depth limit
        max_samples: RF parameter - sample fraction per tree
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (best_n_estimators, convergence_history):
        - best_n_estimators: The number of trees that gave best CV AUC
        - convergence_history: List of dicts with n_estimators, cv_auc, delta at each step
    """
    convergence_history = []
    best_auc = 0.0
    best_n = min_estimators
    no_improvement_count = 0

    for n in range(min_estimators, max_estimators + 1, step):
        cv_aucs = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weight[train_idx] if sample_weight is not None else None

            model = RandomForestClassifier(
                n_estimators=n,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_samples=max_samples,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                oob_score=False,
            )

            model.fit(X_train, y_train, sample_weight=w_train)

            y_proba = model.predict_proba(X_test)

            try:
                # Handle both binary and multi-class cases
                if len(model.classes_) > 2:
                    # Multi-class: use one-vs-rest with weighted average
                    auc = roc_auc_score(y_test, y_proba, multi_class="ovr",
                                       average="weighted", labels=model.classes_)
                else:
                    # Binary: use probability of positive class
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                cv_aucs.append(auc)
            except ValueError:
                # Skip fold if AUC can't be computed (e.g., only one class present)
                pass

        if not cv_aucs:
            logger.warning(f"Early stopping: no valid AUC for n={n}, stopping")
            break

        avg_auc = float(np.mean(cv_aucs))
        delta = avg_auc - best_auc

        convergence_history.append({
            "n_estimators": n,
            "cv_auc": avg_auc,
            "delta": delta,
        })

        logger.info(f"Early stopping: n={n}, CV AUC={avg_auc:.4f}, delta={delta:.4f}")

        if delta > epsilon:
            best_auc = avg_auc
            best_n = n
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            logger.info(
                f"Early stopping: converged at n={best_n} "
                f"(no improvement for {patience} iterations)"
            )
            break

    return best_n, convergence_history
