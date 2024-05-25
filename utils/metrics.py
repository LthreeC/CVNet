from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# specify average here
def calculate_metrics(all_targets, all_predictions, average='macro'):
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average=average)
    f1 = f1_score(all_targets, all_predictions, average=average)

    # Calculate confusion matrix for specificity
    cm = confusion_matrix(all_targets, all_predictions)
    specificity_per_class = []
    for i in range(len(cm)):
        TN = sum(cm[i, j] for j in range(len(cm)) if j != i)
        FP = sum(cm[:, i]) - cm[i, i]
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity_per_class.append(specificity)
    specificity = sum(specificity_per_class) / len(specificity_per_class)

    return accuracy, precision, recall, f1, specificity