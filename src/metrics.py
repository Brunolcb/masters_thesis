import numpy as np


def dice_coef(pred, target, treshold=0.5, accept_zeros = True):
    pred = pred>treshold
    target = target>0.5
    num = 2 * (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        if accept_zeros:
            return 1.0
        else:
            return 0.0
    else:
        return num / denom

def soft_dice(pred, target, eps=1e-12):
    eps = eps
    intersection = np.sum(pred*target)                          
    dice = (2.*intersection + eps)/(np.sum(pred) + np.sum(target) + eps)  
    return dice

def soft_dice_withzeros(pred, target):
    eps = 1e-12
    if np.sum(target) == 0:
        return 1.0
    intersection = np.sum(pred*target)                          
    dice = (2.*intersection + eps)/(np.sum(pred) + np.sum(target) + eps)  
    return dice

def dice_norm_metric(predictions, ground_truth, r = 0.076, threshold = 0.5):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.
    
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Cast to float32 type
    gt = ground_truth>threshold
    gt = gt.astype("float32")
    seg = predictions>threshold
    seg = seg.astype("float32")
    
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm

def soft_dice_norm_metric(pred, target, r=0.0783):
    """
    Compute Normalised Dice Coefficient (nDSC), 
    False positive rate (FPR),
    False negative rate (FNR) for a single example.

    Args:
      target: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      pred:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """
    smooth = 1.

    # Cast to float32 type
    gt = target.astype("float32")
    seg = pred.astype("float32")

    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg * gt)
        fp = np.sum(seg * (1 - gt))
        fn = np.sum((1 - seg) * gt)
        fp_scaled = k * fp
        dsc_norm = (2. * tp + smooth) / (fp_scaled + 2. * tp + fn + smooth)

        return dsc_norm

#função para realizar a curva de retenção
def compute_retention_curve(confidence: np.ndarray, dices: np.ndarray):
    ordering = confidence.argsort()

    scores_ = confidence[ordering]
    dices_ = dices[ordering]

    retention_percentage = list()
    retention_score = list()
    for i in range(len(scores_) + 1):
        retention_percentage.append(i / len(scores_))
        ret_dices_ = np.ones_like(dices_)
        ret_dices_[:i] = dices_[:i]
        retention_score.append(np.mean(ret_dices_))

    return retention_percentage, retention_score

def rc_curve(confidence, dice,expert=True, expert_cost=0):
    error = 1 - dice

    error = np.array(error).reshape(-1)
    confidence = np.array(confidence).reshape(-1)
    n = len(error)
    assert len(confidence) == n
    desc_sort_indices = confidence.argsort()[::-1]
    error = error[desc_sort_indices]
    confidence = confidence[desc_sort_indices]
    idx = np.r_[np.where(np.diff(confidence))[0], n-1]
    thresholds = confidence[idx]
    coverages = (1 + idx)/n
    risks = np.cumsum(error)[idx]/n
    if expert:
        if np.any(expert_cost):
            expert_cost = np.array(expert_cost).reshape(-1)
            if expert_cost.size == 1:
                risks += (1 - coverages)*expert_cost
            else:
                assert len(expert_cost) == n
                expert_cost = np.cumsum(expert_cost)
                expert_cost = expert_cost[-1] - expert_cost
                risks += expert_cost[idx]/n
    else:
        risks /= coverages

    risks = np.insert(risks, 0, risks[0])
    coverages = np.insert(coverages, 0, 0.)
    thresholds = np.insert(thresholds, 0, thresholds[0])
    
    return coverages, risks, thresholds
