from sklearn import metrics

def auc(predictions, true_labels):
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    return auc
