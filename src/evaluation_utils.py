from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def collect_predictions(model, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(predictions.astype(int).tolist())
        y_prob.extend(probabilities.astype(float).tolist())

    if not y_true:
        raise ValueError("No samples were found for evaluation.")

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def save_evaluation_results(
    model,
    dataset,
    class_names: list[str],
    output_dir: Path,
    split_name: str = "test",
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = collect_predictions(model, dataset)
    labels = list(range(len(class_names)))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    report_prefix = output_dir / f"{split_name}_classification_report"
    pd.DataFrame(report_dict).transpose().to_csv(report_prefix.with_suffix(".csv"))
    report_prefix.with_suffix(".txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        output_dir / f"{split_name}_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"{split_name.title()} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{split_name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    probability_df = pd.DataFrame(y_prob, columns=class_names)
    probability_df.insert(0, "true_label", [class_names[index] for index in y_true])
    probability_df.insert(1, "predicted_label", [class_names[index] for index in y_pred])
    probability_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1_score": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1_score": float(macro_f1),
    }

    with open(output_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def print_metrics(metrics: dict[str, float], split_name: str = "test") -> None:
    title = split_name.replace("_", " ").title()
    print(f"\n{title} Metrics")
    print("-" * (len(title) + 8))
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-score:  {metrics['weighted_f1_score']:.4f}")
    print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score:     {metrics['macro_f1_score']:.4f}")
