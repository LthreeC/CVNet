import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc
from plot.config import CLASSES, NUM_CLASSES
from plot.read import read_npy

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


def plot_confusion_matrix(true_labels, predictions, output_path='output/confusion_matrix.svg'):
    cm = confusion_matrix(true_labels, predictions)
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    print(cm_df)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt='g')
    # plt.title(f'{save_name[:-4]}_Confusion Matrix')
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    # plt.savefig(f"output/image/{save_name[:-4]}_confusion_matrix.svg", format='svg')
    plt.savefig(output_path, format='svg')
    plt.show()


# trans it
def plot_multiclass_roc(all_labels, all_prob, output_path="output/multi_roc.svg", nlinewidth=2, nname='None', ncolor='deeppink'):
    all_labels_binarized = label_binarize(all_labels, classes=range(NUM_CLASSES))
    all_prob = np.array(all_prob)

    # calc roc and auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], all_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # use micro
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_binarized.ravel(), np.array(all_prob).ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # draw all class
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"{nname} (auc = {roc_auc['micro']:0.5f})",
             color=ncolor, linestyle='--', linewidth=nlinewidth)

    plt.xlim([-1e-3, 0.5])
    plt.ylim([.8, 1.0 + 1e-3])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")

    plt.savefig(output_path, format='svg')
    plt.show()

def plot_roc_curve_multiclass(y_true, y_pred):
    unique_classes = np.unique(y_true)
    fprs = {}
    tprs = {}
    aucs = {}

    for cls in unique_classes:
        binary_y_true = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(binary_y_true, y_pred)
        fprs[cls] = fpr
        tprs[cls] = tpr
        aucs[cls] = auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], 'k--')
    for cls in unique_classes:
        plt.plot(fprs[cls], tprs[cls], label=f'Class {cls} (AUC = {aucs[cls]:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    true_labels, best_predictions = read_npy("input/Test_Best_targets.npy"), read_npy("input/Test_Best_predictions.npy")
    print(true_labels.shape)
    print(best_predictions.shape)
    plot_confusion_matrix(true_labels, best_predictions)
    plot_roc_curve_multiclass(true_labels, best_predictions)

    # true_labels, final_predictions = read_npy("input/Test_Final_targets.npy"), read_npy("input/Test_Final_predictions.npy")
    # print(true_labels.shape)
    # print(final_predictions.shape)
    # plot_confusion_matrix(true_labels, final_predictions)
    # plot_roc_curve_multiclass(true_labels, final_predictions)