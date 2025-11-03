"""Random Forest Diagnostics - Model Health Checks and Feature Selection

This module helps you understand which features your model actually uses and how well
it's performing. Instead of blindly trusting the model, we run diagnostic tests to
find out what's really going on inside.

Core diagnostics:
- Permutation importance: Measures feature importance by shuffling each feature and
  seeing how much the model's accuracy drops. A big drop means that feature matters.
- Iterative pruning: Removes the least important features one batch at a time and
  retrains to find the optimal feature set. Keeps only what actually helps.
- OOB validation: Tests the model on data it didn't train on by using the max_samples
  parameter to create a bagging effect, similar to out-of-bag scoring.

Technical implementation follows mlbook.explained.ai and fastai best practices using
sklearn.inspection.permutation_importance and cross-validation for robust evaluation.
"""
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def compute_permutation_importances(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[int, float]:
    """Compute feature importance by shuffling each feature and measuring accuracy drop.

    How it works: Take each feature one at a time, randomly shuffle its values
    (breaking any relationship it has with the target), then see how much the model's
    accuracy drops. Features that cause big drops when shuffled are important.

    This is more reliable than tree-based importance (MDI) because it directly measures
    the feature's contribution to predictions rather than just how often it's used.

    Args:
        model: Trained RandomForestClassifier
        X: Feature matrix (samples × features)
        y: Target labels
        n_repeats: How many times to shuffle each feature (higher = more stable results)
        random_state: Random seed for reproducible shuffling

    Returns:
        Dictionary mapping feature index to importance score (higher = more important)
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    return {i: float(result.importances_mean[i]) for i in range(X.shape[1])}


def iterative_feature_pruning(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 500,
    max_features: str = "sqrt",
    min_samples_leaf: int = 50,
    max_samples: float = None,
    n_splits: int = 5,
    min_features: int = 5,
    prune_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[list[str], list[dict]]:
    """Find the optimal feature set by iteratively removing the least important features.

    Process: Start with all features, compute permutation importance, drop the bottom 25%
    (or whatever prune_fraction you set), retrain, and check if performance improved.
    Keep going until you hit min_features or performance starts degrading.

    This is inspired by the fastai tabular approach: often you can remove a lot of
    low-importance features and actually improve model performance (less noise, less
    overfitting, faster training).

    Args:
        X: Feature matrix (samples × features)
        y: Target labels
        feature_names: Names of each feature column
        n_estimators: Number of trees in the forest
        max_features: Max features to consider per split ("sqrt", "log2", or float)
        min_samples_leaf: Minimum samples required in each leaf node
        max_samples: Fraction of samples per tree (enables OOB-like validation)
        n_splits: Number of cross-validation folds to evaluate each iteration
        min_features: Stop pruning when this many features remain
        prune_fraction: What fraction of remaining features to drop each iteration
        random_state: Random seed for reproducible results

    Returns:
        Tuple of (best_features, pruning_history):
        - best_features: List of feature names in the best-performing subset
        - pruning_history: List of dicts with CV metrics for each iteration
    """
    current_features = list(feature_names)
    current_X = X.copy()
    pruning_history = []
    best_score = -np.inf
    best_features = current_features.copy()

    iteration = 0
    while len(current_features) > min_features:
        logger.info(f"Pruning iteration {iteration}: {len(current_features)} features")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            oob_score=True if max_samples is not None else False,
        )

        scores = cross_val_score(
            model, current_X, y, cv=n_splits, scoring="roc_auc", n_jobs=-1
        )
        cv_score = float(np.mean(scores))

        model.fit(current_X, y)

        perm_importances = compute_permutation_importances(
            model, current_X, y, n_repeats=5, random_state=random_state
        )

        iteration_data = {
            "iteration": iteration,
            "n_features": len(current_features),
            "cv_auc": cv_score,
            "cv_std": float(np.std(scores)),
            "features": current_features.copy(),
        }

        if max_samples is not None:
            iteration_data["oob_score"] = float(model.oob_score_)

        pruning_history.append(iteration_data)

        if cv_score > best_score:
            best_score = cv_score
            best_features = current_features.copy()
            logger.info(
                f"  New best CV AUC: {cv_score:.4f} with {len(best_features)} features"
            )

        n_to_drop = max(1, int(len(current_features) * prune_fraction))

        if len(current_features) - n_to_drop < min_features:
            break

        sorted_indices = sorted(
            perm_importances.items(), key=lambda x: x[1], reverse=False
        )
        drop_indices = [idx for idx, _ in sorted_indices[:n_to_drop]]

        current_features = [
            f for i, f in enumerate(current_features) if i not in drop_indices
        ]
        feature_indices = [
            i for i in range(current_X.shape[1]) if i not in drop_indices
        ]
        current_X = current_X[:, feature_indices]

        iteration += 1

    logger.info(
        f"Pruning complete: best {len(best_features)} features with AUC {best_score:.4f}"
    )

    return best_features, pruning_history


def compute_oob_metrics(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    max_features: str = "sqrt",
    min_samples_leaf: int = 50,
    max_samples: float = 0.7,
    random_state: int = 42,
) -> dict:
    """Check for overfitting by comparing training score to out-of-bag score.

    When max_samples < 1.0, each tree only sees a random subset of the data. The
    samples not used to train a tree can be used to test it, giving us an "out-of-bag"
    score. This lets us estimate validation performance without a separate validation set.

    The train/val gap is critical: if training score is much higher than OOB score,
    your model is overfitting. mlbook.explained.ai emphasizes watching this gap.

    Args:
        X: Feature matrix (samples × features)
        y: Target labels
        n_estimators: Number of trees in the forest
        max_features: Max features to consider per split
        min_samples_leaf: Minimum samples required in each leaf node
        max_samples: Fraction of samples each tree trains on (e.g., 0.7 = 70%)
        random_state: Random seed for reproducible results

    Returns:
        Dictionary with keys:
        - oob_score: Performance on out-of-bag samples
        - train_score: Performance on training data
        - train_val_gap: Difference between train and OOB (watch for overfitting)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )

    model.fit(X, y)

    train_score = float(model.score(X, y))
    oob_score = float(model.oob_score_)
    gap = train_score - oob_score

    return {
        "oob_score": oob_score,
        "train_score": train_score,
        "train_val_gap": gap,
    }


def generate_diagnostics_report(
    permutation_importances: dict[int, float],
    feature_names: list[str],
    pruning_history: list[dict] = None,
    oob_metrics: dict = None,
) -> dict:
    """Package all diagnostic results into a JSON-safe report for storage.

    Combines permutation importances, feature pruning results, and OOB validation
    metrics into a single structured dictionary that can be saved to MLRun metadata
    or written to a file.

    Args:
        permutation_importances: Feature index → importance score mapping
        feature_names: List of feature names matching indices
        pruning_history: Results from iterative_feature_pruning (optional)
        oob_metrics: Results from compute_oob_metrics (optional)

    Returns:
        JSON-serializable dict with keys:
        - permutation_importances: top_20 and bottom_20 features
        - pruning: iteration history and best iteration (if provided)
        - oob_validation: train/val gap metrics (if provided)
    """
    sorted_importances = sorted(
        [
            (feature_names[idx], importance)
            for idx, importance in permutation_importances.items()
            if idx < len(feature_names)
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    report = {
        "permutation_importances": {
            "top_20": [
                {"feature": name, "importance": float(imp)}
                for name, imp in sorted_importances[:20]
            ],
            "bottom_20": [
                {"feature": name, "importance": float(imp)}
                for name, imp in sorted_importances[-20:]
            ],
        }
    }

    if pruning_history:
        report["pruning"] = {
            "iterations": pruning_history,
            "best_iteration": max(pruning_history, key=lambda x: x["cv_auc"]),
        }

    if oob_metrics:
        report["oob_validation"] = oob_metrics

    return report
