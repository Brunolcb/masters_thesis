from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import auc

from src.metrics import rc_curve, dice_coef, hd95


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
    '''random_risks = np.zeros(scores.shape)
    n_random = 10
    for _ in range(n_random):
        coverages, risks, _ = rc_curve(np.random.rand(*scores.shape),
                                       scores, expert=False, expert_cost=0, type_dice=type_dice)
        random_risks += risks / n_random'''
    if type_dice == True:
        random_risks = np.repeat(1-np.mean(scores), scores.shape)
        coverages = np.linspace(0,1,scores.shape[0])
    else:
        random_risks = np.repeat(np.mean(scores), scores.shape)
        coverages = np.linspace(0,1,scores.shape[0])
        
    rcauc = auc(coverages, random_risks)
    plt.plot(coverages, random_risks, linestyle='--', label=f"random ({rcauc:.3f})")

    ylim = list(plt.ylim())
    ylim[0] = 0.
    plt.ylim(ylim)
    plt.xlim(0,1)
    plt.legend()

def plot_segmentation_performance_report(results_fpath):
    _results_fpath = Path(results_fpath)

    data = np.load(_results_fpath)
    y = data['y']
    y_hat = data['y_hat']

    assert np.equal(y.shape, y_hat.shape).all()

    hd95s = list(map(
        lambda ys_i: hd95(ys_i[0].squeeze(), ys_i[1].squeeze()),
        zip(y, y_hat)
    ))

    dices = list(map(
        lambda ys_i: dice_coef(ys_i[0].flatten(), ys_i[1].flatten()),
        zip(y, y_hat)
    ))

    fig, axs = plt.subplots(3,1)
    fig.set_size_inches(8,10)
    fig.suptitle(results_fpath.name)

    def plot_performance_hist(values, ax):
        ax.hist(values, bins=20)
        ylims = ax.get_ylim()
        mean_performance = np.mean(values)
        ax.vlines(mean_performance, *ylims, color='red', label=f"{mean_performance:.2f}")
        ax.set_ylim(*ylims)

        return ax

    axs[0].remove()
    ax_hd95 = plot_performance_hist(hd95s, fig.add_subplot(3,2,1))
    ax_hd95.set_xlabel('Hausdorff95')
    ax_hd95.legend()

    ax_dice = plot_performance_hist(dices, fig.add_subplot(3,2,2))
    ax_dice.set_xlabel('Dice')
    ax_dice.legend()

    gt_sizes = y.reshape(y.shape[0],-1).sum(1)
    # axs[1].hist(gt_sizes, bins=[0,]+np.linspace(1,max(gt_sizes),20,endpoint=True),
    axs[1].hist(gt_sizes, bins=20,
                label=f"# of empty = {np.sum(np.array(gt_sizes) == 0)}")
    axs[1].set_yscale('log')
    axs[1].legend()
    axs[1].set_xlabel('gt size')

    probs = y_hat.flatten()
    axs[2].hist(probs, bins=20)
    axs[2].set_yscale('log')
    axs[2].set_xlabel('y_hat probability')

    fig.tight_layout()

    return fig
