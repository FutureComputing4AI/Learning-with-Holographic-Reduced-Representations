"""
Library functions to compute different metrics for tasks.
"""

__author__ = "Ashwinkumar Ganesan"
__email__ = "gashwin1@umbc.edu"

from tabulate import tabulate
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
import xclib.evaluation.xc_metrics as xc_metrics

# Compute the precision score for multi-label binary classification task.
def mbprecision(y_true, y_pred):
    correct_pred = torch.sum(y_pred & y_true, axis=1).float()
    print(correct_pred.dtype)
    return torch.mean(correct_pred / torch.sum(y_true, axis=1))

# Compute the recall score for multi-label binary classification task.
def mbrecall(y_true, y_pred):
    return torch.mean(torch.sum(y_pred & y_true, axis=1) / torch.sum(y_true, axis=1))


def plot_tr_stats(tr_stats, th_stats, spoch, sth, filename):
    """
    Plot stats about the experiment.
    tr_stats: Training statistics (includes loss, precision, recall and F1)
    th_stats: Grid search statistics for configuring threshold.
    epochs: Number of epochs that the model is trained for.
    spoch: epoch that has optimal paramaters.
    sth: optimal threshold.
    filename: location to store plots.
    """
    fig, ax = plt.subplots(3, figsize=(10, 10))

    ep = tr_stats['Epoch']
    tr_loss = tr_stats['Training Loss']
    val_loss = tr_stats['Val Loss']
    pr = tr_stats['Precision']
    re = tr_stats['Recall']
    f1 = tr_stats['F1 Score']
    th = th_stats['Threshold']

    ax[0].plot(ep, tr_loss)
    ax[0].plot(ep, val_loss)
    ax[0].set_title("Training & Validation Loss Per Epoch", size=16)
    ax[0].set_xlabel("Epoch", size=14)
    ax[0].set_ylabel("Loss", size=14)
    ax[0].legend(["Training Loss", "Validation Loss"], fontsize="large")
    ax[0].axvline(x=spoch, linestyle='dashed')

    ax[1].plot(ep, pr)
    ax[1].plot(ep, re)
    ax[1].plot(ep, f1)
    ax[1].set_title("Validation Precision, Recall & F-1 Score \n (Threshold = 0.25)", size=16)
    ax[1].set_xlabel("Epoch", size=14)
    ax[1].set_ylabel("Score", size=14)
    ax[1].legend(["Validation Precision", "Validation Recall", "Validation F1 Score"], fontsize="large")
    ax[1].axvline(x=spoch, linestyle='dashed')

    ax[2].plot(th, th_stats['Precision'])
    ax[2].plot(th, th_stats['Recall'])
    ax[2].plot(th, th_stats['F1 Score'])
    ax[2].set_title("Validation Precision, Recall & F-1 Score \n Optimize Threshold", size=16)
    ax[2].set_xlabel("Theshold", size=14)
    ax[2].set_ylabel("Score", size=14)
    ax[2].legend(["Validation Precision", "Validation Recall", "Validation F1 Score"], fontsize="large")
    ax[2].axvline(x=sth, linestyle='dashed')

    fig.tight_layout()
    plt.savefig(filename + ".png")

# Adapted from: https://github.com/kunaldahiya/pyxclib
def compute_inv_propensity(train_labels, A=0.55, B=1.5):
    """
        Compute Inverse propensity values
        Values for A/B:
            Wikpedia-500K: 0.5/0.4
            Amazon-670K, Amazon-3M: 0.6/2.6
            Others: 0.55/1.5

        Arguments:
        train_labels : numpy ndarray
    """
    inv_propen = xc_metrics.compute_inv_propesity(train_labels, A, B)
    return inv_propen

# Compute metrics with propensity.
def compute_prop_metrics(true_labels, predicted_labels, inv_prop_scores, topk=5):
    """Compute propensity weighted precision@k and DCG@k.
       Arguments:
       true_labels : numpy ndarray
                     Ground truth labels from the dataset (one-hot vector).
       predicted_labels : numpy ndarray
                          Predicted labels (one-hot vector of labels)
    """
    acc = xc_metrics.Metrics(true_labels=true_labels, inv_psp=inv_prop_scores,
                             remove_invalid=False)
    return acc.eval(predicted_labels, topk)

# Print the final results.
# This provides the results for agg metrics when threshold for inference
# is optimized and metrics are then computed.
def display_agg_results(args, te_loss, pr, rec, f1):
    print("----------Tests with Threshold Inference------------")
    print("Inference Threshold: {:.3f}".format(args.th))
    print("Test Loss: {:.3f}".format(te_loss))
    print("Test Precision: {:.3f}".format(pr * 100))
    print("Test Recall: {:.3f}".format(rec * 100))
    print("Test F1-Score: {:.3f}\n".format(f1 * 100))


def display_metrics(metrics, k=5):
    # Merge batchwise metrics.
    final_metrics = [[0.0] * k,[0.0] * k,[0.0] * k,[0.0] * k]
    for idx, metric in enumerate(metrics):
        for i in range(0, 4):
            for j in range(0, k):
                final_metrics[i][j] += metric[i][j]

    # Dataset metrics.
    print("----------Tests with Ordered Retrieval------------")
    table = [['Precision@k'] + [i * 100 / (idx + 1) for i in final_metrics[0]]]
    table.append(['nDCG@k'] + [i * 100 / (idx + 1) for i in final_metrics[1]])
    table.append(['PSprec@k'] + [i * 100 / (idx + 1) for i in final_metrics[2]])
    table.append(['PSnDCG@k'] + [i * 100 / (idx + 1) for i in final_metrics[3]])
    print(tabulate(table, headers=[i+1 for i in range(0, k)],
                   floatfmt=".3f"))
