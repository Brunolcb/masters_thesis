import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import auc

from src.metrics import rc_curve


def plot_rc_curves(uncertainties: dict, scores, type_dice=True):
    for metric, uncertainty in uncertainties.items():
        coverages, risks, _ = rc_curve(-uncertainty, scores, expert=False, expert_cost=0, type_dice=type_dice)
        rcauc = auc(coverages, risks)
        plt.plot(coverages, risks, label=f"{metric} ({rcauc:.3f})")

    #ideal RC curve
    if type_dice == True:
        coverages, risks, _ = rc_curve(scores, scores, expert=False, expert_cost=0)
    else:
        coverages, risks, _ = rc_curve(-scores, scores, expert=False, expert_cost=0, type_dice=type_dice)
    rcauc = auc(coverages, risks)
    plt.plot(coverages, risks, linestyle='--', label=f"ideal ({rcauc:.3f})")

    # random uncertainty estimation
    random_risks = np.zeros(scores.shape)
    n_random = 10
    for _ in range(n_random):
        coverages, risks, _ = rc_curve(np.random.rand(*scores.shape),
                                       scores, expert=False, expert_cost=0, type_dice=type_dice)
        random_risks += risks / n_random
    rcauc = auc(coverages, random_risks)
    plt.plot(coverages, random_risks, linestyle='--', label=f"random ({rcauc:.3f})")

    ylim = list(plt.ylim())
    ylim[0] = 0.
    plt.ylim(ylim)
    plt.xlim(0,1)
    plt.legend()