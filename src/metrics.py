import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

def sigmoid(x, alpha, beta, gamma):
    return gamma/ (1 + np.exp(-alpha*x+beta))

def dice_coef(pred, target, treshold=0.5):
    pred = pred>treshold
    target = target>0.5
    num = 2 * (pred & target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    else:
        return num / denom

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

def soft_dice(pred, target, eps=1e-12):
    eps = eps
    intersection = np.sum(pred*target)                          
    dice = (2.*intersection + eps)/(np.sum(pred) + np.sum(target) + eps)  
    return dice

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

def rc_curve(confidence, metric,expert=True, expert_cost=0, type_dice =True):
    if type_dice == True:
        error = 1 - metric
    else:
        error = metric
        
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
        
    if len(risks)!=n:
        coverages = np.insert(coverages, 0, 0)
        risks = np.insert(risks, 0, 0)
        thresholds = np.insert(thresholds, 0, 0)
    
    return coverages, risks, thresholds



def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds