"""
Evaluation metrics shared by all models, suited to imbalanced 4-class labels.

Accuracy alone is misleading here (Stage 6 is ~52% of cells, so always
predicting Stage 6 scores ~52%). We report balanced accuracy, macro-F1 and
per-class recall as the headline numbers, plus donor-level accuracy
(average the predicted probabilities of each test donor's cells).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score, confusion_matrix)

STAGE_NAMES = ['Stage 1', 'Stage 2', 'Stage 5', 'Stage 6']
LABELS = [0, 1, 2, 3]


def compute_metrics(y_true, y_pred, probs=None, donors=None):
    """
    Parameters
    ----------
    y_true, y_pred : array (n_cells,)
    probs : array (n_cells, 4) or None - class probabilities, needed for AUC
            and donor-level accuracy
    donors : array (n_cells,) or None - donor ID per cell

    Returns
    -------
    dict of JSON-serialisable metrics
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        'n_cells': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)),
        'per_class_recall': {
            s: float(r) for s, r in zip(
                STAGE_NAMES, recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0))
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }

    if probs is not None:
        probs = np.asarray(probs)
        present = [c for c in LABELS if (y_true == c).any()]
        # One-vs-rest AUC only for classes present in this test fold
        aucs = {}
        for c in present:
            if len(np.unique(y_true == c)) == 2:
                aucs[STAGE_NAMES[c]] = float(roc_auc_score(y_true == c, probs[:, c]))
        out['per_class_auc'] = aucs
        out['macro_auc'] = float(np.mean(list(aucs.values()))) if aucs else None

    if probs is not None and donors is not None:
        donors = np.asarray(donors)
        d_true, d_pred = [], []
        for d in np.unique(donors):
            m = donors == d
            d_true.append(int(y_true[m][0]))
            d_pred.append(int(probs[m].mean(axis=0).argmax()))
        out['donor_level'] = {
            'n_donors': len(d_true),
            'accuracy': float(accuracy_score(d_true, d_pred)),
            'true': d_true,
            'pred': d_pred,
        }
    return out


def majority_baseline(y_train, y_test, donors_test=None):
    """Always predict the most common training class."""
    majority = int(np.bincount(y_train, minlength=4).argmax())
    y_pred = np.full(len(y_test), majority)
    probs = np.zeros((len(y_test), 4))
    probs[:, majority] = 1.0
    return compute_metrics(y_test, y_pred, probs, donors_test)
