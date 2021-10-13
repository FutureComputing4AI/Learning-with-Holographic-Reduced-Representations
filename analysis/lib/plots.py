"""
Manage plots.
AUTHOR: Ashwinkumar Ganesan.
"""

import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas as pd

"""
Plot training and testing curves.
The graph includes:
1. Training loss per epoch.
2. Test loss per epoch.
3. Precision per epoch.
4. Recall per epoch.
5. F1 score per epoch.
"""
def plot_stats(tr_stats):    
    fig, ax = plt.subplots(2)
    
    ep = [i for i in range(0, epochs)]
    tr_loss = tr_stats['Training Loss']
    te_loss = tr_stats['Test Loss']
    pr = tr_stats['Precision']
    re = tr_stats['Recall']
    f1 = tr_stats['F1 Score']
    
    # Loss Curve.
    ax[0].plot(ep, tr_loss)
    ax[0].plot(ep, te_loss)
    ax[0].set_title("Training & Testing Loss Per Epoch")
    
    
    ax[1].plot(ep, pr)
    ax[1].plot(ep, re)
    ax[1].plot(ep, f1)
