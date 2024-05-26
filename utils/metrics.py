from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# specify average here
# def calculate_metrics(all_targets, all_predictions, average='macro'):
#     accuracy = accuracy_score(all_targets, all_predictions)
#     precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
#     recall = recall_score(all_targets, all_predictions, average=average)
#     f1 = f1_score(all_targets, all_predictions, average=average)
#
#     # Calculate confusion matrix for specificity
#     cm = confusion_matrix(all_targets, all_predictions)
#     specificity_per_class = []
#     for i in range(len(cm)):
#         TN = sum(cm[i, j] for j in range(len(cm)) if j != i)
#         FP = sum(cm[:, i]) - cm[i, i]
#         specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
#         specificity_per_class.append(specificity)
#     specificity = sum(specificity_per_class) / len(specificity_per_class)
#
#     return accuracy, precision, recall, f1, specificity

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score


def calculate_metrics(all_targets, all_predictions, average='macro'):
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average=average, zero_division=0)
    recall = recall_score(all_targets, all_predictions, average=average)
    f1 = f1_score(all_targets, all_predictions, average=average)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # Specificity
    TN_sum = 0
    FP_sum = 0
    for i in range(len(cm)):
        TN = sum(cm[j, j] for j in range(len(cm)) if j!= i)
        FP = sum(cm[:, i]) - cm[i, i]
        TN_sum += TN
        FP_sum += FP
    specificity = TN_sum / (TN_sum + FP_sum) if (TN_sum + FP_sum) > 0 else 1

    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_targets, all_predictions, average=average)
    except ValueError:
        roc_auc = 0

    # Average Precision
    try:
        average_precision = average_precision_score(all_targets, all_predictions)
    except ValueError:
        average_precision = 0

    return accuracy, precision, recall, f1, specificity, roc_auc, average_precision