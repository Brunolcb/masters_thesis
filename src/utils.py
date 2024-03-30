from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
import pandas as pd
from sklearn.metrics import auc

from src.metrics import rc_curve, dice_coef, hd95


def get_noise_with_Lcov(Lcov_path: str, img_shape: tuple, n=None):
    L = da.from_npy_stack(Lcov_path)

    if n is None:
        da.random.seed(0)
        eta = da.random.normal(0, size=L.shape[1])
        eta = L @ eta
        eta = eta.reshape(*img_shape)
    else:
        da.random.seed(0)
        eta = da.random.normal(0, size=(n, L.shape[1]))
        eta = eta @ L.T
        eta = eta.reshape(n, *img_shape)

    return eta.compute()

def plot_rc_curves(confidences: dict, errors: np.array, ax):
    random_aurc = np.mean(errors)

    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)

    ax = plot_baselines(errors, ax)

    for name, confidence in confidences.items():
        plot_rc_curve(confidence, errors, name, ax, low_aurc=ideal_aurc, high_aurc=random_aurc)

    ax.set_xlim(0,1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid()
    ax.legend()

    return ax

def plot_baselines(errors, ax, **kwargs):
    ax.hlines(np.mean(errors), 0, 1, colors='gray', linestyles='dashed', **kwargs)

    coverages, risks, _ = rc_curve(-errors, errors, expert=False, ideal=True)

    ax.plot(coverages, risks, linestyle='dashed', c='gray', **kwargs)

    return ax

def plot_rc_curve(confidence, errors, label, ax, low_aurc=0, high_aurc=None,
                  **kwargs):
    coverages, risks, _ = rc_curve(confidence, errors)

    aurc = auc(coverages, risks)
    aurc -= low_aurc
    
    if high_aurc is not None:
        high_aurc -= low_aurc
        aurc = aurc / high_aurc
     
    ax.plot(coverages, risks, label=f"{label}")
    #ax.plot(coverages, risks, label=f"{label} ({aurc:.3f})")

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


def plot_aurc_curves(confidences: dict, errors: np.array, ax):
    random_aurc = np.mean(errors)

    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)

    ax = plot_baselines(errors, ax)

    for name, confidence in confidences.items():
        plot_rc_curve(confidence, errors, name, ax)

    ax.set_xlim(0,1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.grid()
    ax.legend()

    return ax

def write_aurc_curves(confidences: dict, errors: np.array):
    aurcs = {}
    random_aurc = np.mean(errors)

    ideal_coverage, ideal_risk, _ = rc_curve(-errors, errors, ideal=True)
    ideal_aurc = auc(ideal_coverage, ideal_risk)
    aurcs['Random'] = random_aurc
    aurcs['Ideal'] = ideal_aurc
    for name, confidence in confidences.items():
        coverage, risk, _ = rc_curve(confidence, errors)
        aurc = auc(coverage, risk)
        aurcs[name] = aurc
    return aurcs

def calculating_aurc_from_dataframe(dataframe: pd.DataFrame, risks =['dice risk', 'ndice risk','hd95 risk']):
    ideal_coverage = {}
    ideal_risk = {}
    aurcs = {}
    errors = {name:dataframe[name] for name in risks}
    confidences =  dataframe.drop(risks, axis=1)
    errors = {name:dataframe[name] for name in risks}
    random_aurc = {name: np.mean(errors[name]) for name in errors.keys()}

    for name in risks:
        ideal_coverage[name], ideal_risk[name], _ = rc_curve(-errors[name], errors[name], ideal=True)
        ideal_aurc = auc(ideal_coverage[name], ideal_risk[name])
        aurcs[f'Random_{name}'] = random_aurc[name]
        aurcs[f'Ideal_{name}'] = ideal_aurc
        for name_conf, confidence in confidences.items():
            coverage, risk, _ = rc_curve(confidence, errors[name])
            aurc = auc(coverage, risk)
            aurcs[name_conf+'_'+name] = aurc
    return aurcs

