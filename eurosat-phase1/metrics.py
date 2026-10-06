from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score


def classification_metrics(targets, predictions, classes):
    labels = list(range(len(classes)))
    report = classification_report(targets, predictions, labels=labels,
                                   target_names=classes, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(targets, predictions)),
            'macro_f1': float(f1_score(targets, predictions, labels=labels,
                                     average='macro', zero_division=0)),
            'balanced_accuracy': float(balanced_accuracy_score(targets, predictions)),
            'per_class_recall': {name: report[name]['recall'] for name in classes},
            'classification_report': report}
