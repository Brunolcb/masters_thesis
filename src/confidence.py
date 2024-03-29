from abc import ABC, abstractmethod
from typing import Any, Callable
from multiprocessing import Pool
import numpy as np
from scipy.special import softmax, logit, expit

from src.metrics import dice_norm_metric, dice_coef, soft_dice, sigmoid, soft_dice_norm_metric, hd95


def _add_noise_get_dice(args):
    p, sigma = args
    p_eta = expit(
        logit(p) + np.random.normal(0, sigma) * np.ones_like(p)
    )
    return dice_coef(p_eta, p)

class SegmentationConfidence(ABC):
    """Apply confidence metric to image.
    """
    @abstractmethod
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """

    def __call__(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        if probs.shape[0] == 1:  # binary classification
            background = 1. - probs[0]
            probs = np.stack([background, probs[0]], axis=0)

        return self.metric(probs)
    
class SegmentationConfidenceWithNoise(ABC):
    """Apply uncertainty metric to image.
    """
    @abstractmethod
    def metric(self, probs: np.array, probs_with_noise: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """

    def __call__(self, probs: np.array, probs_with_noise: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        if probs.shape[0] == 1:  # binary classification
            background = 1. - probs[0]
            probs = np.stack([background, probs[0]], axis=0)
            
        if probs_with_noise.shape[0] == 1:  # binary classification
            background_with_noise = 1. - probs_with_noise[0]
            probs_with_noise = np.stack([background_with_noise, probs_with_noise[0]], axis=0)
            
        return self.metric(probs, probs_with_noise)

class MeanMaxConfidence(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        confidence = np.max(probs, axis=0)
        return np.mean(confidence)

class ForegroundMeanMaxConfidence(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        confidence = np.max(probs, axis=0)
        confidence *= p >= .5

        return np.mean(confidence)

class ExpectedEntropy(SegmentationConfidence):
    def metric(self, probs: np.array, epsilon=1e-9) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        log_probs = np.log(probs + epsilon)
        exe = -np.sum(probs * log_probs, axis=0)  # sums over classes

        return -np.mean(exe)

class PredSize(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        assert probs.shape[0] == 2

        pred = np.argmax(probs, axis=0)

        return np.sum(pred)

class PredSizeAtThreshold(SegmentationConfidence):
    def __init__(self, threshold=.99):
        super().__init__()
        self.threshold = threshold
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        return np.sum(p > self.threshold)

class PredSizeChange(SegmentationConfidence):
    def __init__(self, thresholds=(.05, .95)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        pred_low = p >= self.thresholds[0]
        pred_high = p >= self.thresholds[1]

        return -np.sum(pred_low ^ pred_high)

class PredChangenDSC(SegmentationConfidence):
    def __init__(self, thresholds=(.1, .9)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        pred_low = p >= self.thresholds[0]
        pred_high = p >= self.thresholds[1]

        return dice_norm_metric(pred_low, pred_high)

class PredChangeDSC(SegmentationConfidence):
    def __init__(self, thresholds=(.1, .9)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        pred_low = p >= self.thresholds[0]
        pred_high = p >= self.thresholds[1]

        return dice_coef(pred_low, pred_high)

class nDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        if np.sum(pred)>0:
            ndscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                ndscs += dice_norm_metric(p > threshold, pred)

            return ndscs
        else:
            return -np.inf

class DSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += dice_coef(p > threshold, pred)

            return dscs
        else:
            return -np.inf

    
class SoftDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        dscs = 0
        for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
            dscs += soft_dice(p,p > threshold)

        return dscs

class NewDSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        if np.sum(pred)>0:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                dscs += dice_coef(p > threshold, pred)
        else:
            dscs = 0
            for threshold in np.arange(self.threshold_lims[0]/2, self.threshold_lims[1]/2 + self.step/2, self.step/2):
                dscs += dice_coef(p > threshold, pred)
        return dscs

class BinaryCrossEntropy(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0) 
        y_true = p > self.threshold_lim
        y_pred = np.clip(p, 1e-7, 1 - 1e-7)
        term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
        term_1 = y_true * np.log(y_pred + 1e-7)
        
        return np.mean(term_0+term_1)

class FocalLoss(SegmentationConfidence):
    def __init__(self, threshold_lim=.5, gamma=2):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.gamma =gamma
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim
        eps = 1e-12
        # If actual value is true, keep pt value as y_pred otherwise (1-y_pred)
        pt = np.where(y_true == 1, y_pred, 1-y_pred)
        # Clip values below epsilon and above 1-epsilon
        pt = np.clip(pt, eps, 1-eps)
        # FL = -(1-pt)^gamma log(pt)
        focal_loss = -np.mean(np.multiply(np.power(1-pt,self.gamma),np.log(pt)))    
        return -focal_loss   
    
class SoftDice(SegmentationConfidence):
    def __init__(self, threshold_lim=.5,eps=1e-12):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps=eps
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        dice = soft_dice(y_pred,y_true,eps=self.eps)  
        return dice   

class SoftDiceWithNoise(SegmentationConfidenceWithNoise):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array, probs_with_noise: np.array) -> float:
        y_true = np.sum(probs[1:], axis=0) 
        y_true = y_true > self.threshold_lim
        y_pred = np.sum(probs_with_noise[1:], axis=0)
        dice = soft_dice(y_pred,y_true)  
        return dice  
    
class DSCIntegralOverThresholdWithNoise(SegmentationConfidenceWithNoise):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    
    def metric(self, probs: np.array, probs_with_noise: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pwn = np.sum(probs_with_noise[1:], axis=0)  # probability of foreground
        gt = p > .5

        dscs = 0
        for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
            dscs += dice_coef(pwn > threshold, gt)

        return dscs

class DSCIntegralOverNoise(SegmentationConfidence):
    def __init__(self, N: int, sigma: float) -> None:
        super().__init__()

        self.N = N
        self.sigma = sigma
    
    def metric(self, probs: np.array) -> float:
        p = np.sum(probs[1:], axis=0)  # probability of foreground

        dscs = list()
        with Pool(10) as pool:
            dscs = pool.map(_add_noise_get_dice, [(p, self.sigma),] * self.N)

        return np.mean(dscs)

class DSCIntegralOverMaxThreshold(SegmentationConfidence):
    def __init__(self, quant_step=20):
        super().__init__()
        self.quant_step = quant_step
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        thresh_lim = np.max(p)
        pred = p > .5

        dscs = []
        for threshold in np.linspace(0.0,thresh_lim,20)[1:]:
            dscs.append(dice_coef(p > threshold, pred))

        return np.mean(dscs)
    
class SoftlogDice(SegmentationConfidence):
    def __init__(self, threshold_lim=.5):
        super().__init__()
        self.threshold_lim = threshold_lim
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim  
        y_predlog = np.where(y_pred>0.5,y_pred*(1+np.log(y_pred)),(y_pred)*(1+np.log(1-y_pred)))
        dice = soft_dice(y_predlog,y_true)  
        return dice   

class SoftDice_Var_Eps(SegmentationConfidence):
    def __init__(self, threshold_lim=.5, quant=10):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.quant = quant
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0)
        y_true = y_pred > self.threshold_lim
        if np.sum(y_true)>0:
            dice = soft_dice(y_pred,y_true) 
            return dice
        else:
            max_logit = np.max(y_pred)
            dice = soft_dice(y_pred,y_true, eps=(1-max_logit)*self.quant)
            return dice      

class DSCIntegralOverThreshold_bins(SegmentationConfidence):
    def __init__(self, threshold_lims_sup=np.array([0.25,0.5,0.75]), threshold_lims_inf=np.array([0.125,0.25,0.5])):
        super().__init__()
        self.threshold_lims_sup = threshold_lims_sup
        self.threshold_lims_inf = threshold_lims_inf

    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        dscs = 0
        if np.sum(pred)>0:
            for threshold in self.threshold_lims_sup:
                dscs += dice_coef(p > threshold, pred)
                return dscs
        else:
            for threshold in self.threshold_lims_inf:
                dscs += dice_coef(p > threshold, pred)
                return dscs

class Soft_Dice_and_DSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims_sup=np.array([0.25,0.5,0.75]), threshold_lims_inf=np.array([0.125,0.25,0.5])):
        super().__init__()
        self.threshold_lims_sup = threshold_lims_sup
        self.threshold_lims_inf = threshold_lims_inf

    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        dscs = 0
        if np.sum(pred)>0:
            for threshold in self.threshold_lims_sup:
                dscs += dice_coef(p > threshold, pred)
                return dscs
        else:
            for threshold in self.threshold_lims_inf:
                dscs += dice_coef(p > threshold, pred)
                return dscs
            
class Soft_Dice_and_DSCIntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, threshold=.5,eps=1e-12):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.eps = eps
        self.threshold = threshold
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > self.threshold 
        if np.sum(pred)>0:
            dscs = 0
            interval = np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step)
            for threshold in interval:
                dscs += dice_coef(p > threshold, pred)
            dscs = dscs/(interval.shape[0])
            return dscs
        else:
            dice = soft_dice(p,pred,eps=self.eps)
            return dice

class DIOT_Sigmoid(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05, threshold=.5, alpha=1.0, beta= 0.0, gamma=1.0):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
        self.alpha = alpha 
        self.beta = beta
        self.threshold = threshold
        self.gamma = gamma
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > self.threshold 
        if np.sum(pred)>0:
            dscs = 0
            interval = np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step)
            for threshold in interval:
                dscs += dice_coef(p > threshold, pred)
            dscs = dscs/(interval.shape[0])
            return dscs
        else:
            return sigmoid(np.sum(p), self.alpha, self.beta,self.gamma)
        
class ExpectedEntropy_With_If(SegmentationConfidence):
    def metric(self, probs: np.array, epsilon=1e-9) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > 0.5
        if np.sum(pred)>0:
            log_probs = np.log(probs + epsilon)
            exe = -np.sum(probs * log_probs, axis=0)  # sums over classes

            return -np.mean(exe)           
        else:
            return -np.inf
        
class PredChangeDSC_With_If(SegmentationConfidence):
    def __init__(self, thresholds=(.1, .9)):
        super().__init__()
        self.thresholds = thresholds
    
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > 0.5
        if np.sum(pred)>0:
            
            pred_low = p >= self.thresholds[0]
            pred_high = p >= self.thresholds[1]

            return dice_coef(pred_low, pred_high)   
        else:
            return -np.inf
        
class MeanMaxConfidence_With_If(SegmentationConfidence):
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > 0.5
        if np.sum(pred)>0:
            confidence = np.max(probs, axis=0)
            return np.mean(confidence)
        else:
            return -np.inf

class SoftnDice(SegmentationConfidence):
    def __init__(self, threshold_lim=.5,eps=1e-12, r=0.0783):
        super().__init__()
        self.threshold_lim = threshold_lim
        self.eps=eps
        self.r = r
    def metric(self, probs: np.array) -> float:
        y_pred = np.sum(probs[1:], axis=0) 
        y_true = y_pred > self.threshold_lim        
        ndice = soft_dice_norm_metric(y_pred,y_true,r=self.r)  
        return ndice   
    
class HD95IntegralOverThreshold(SegmentationConfidence):
    def __init__(self, threshold_lims=(.05, .95), step=.05):
        super().__init__()
        self.threshold_lims = threshold_lims
        self.step = step
    def metric(self, probs: np.array) -> float:
        """
        :param probs: array [num_classes, *image_shape]
        :return: float
        """
        p = np.sum(probs[1:], axis=0)  # probability of foreground
        pred = p > .5
        
        if np.sum(pred)>0:
            hd95_value = 0
            count =0
            for threshold in np.arange(self.threshold_lims[0], self.threshold_lims[1] + self.step, self.step):
                if np.sum(p > threshold)>0:
                    hd95_value += hd95(p > threshold, pred)
                    count +=1
                
            return -hd95_value/count
        else:
            return np.inf
#class SquareDistances(SegmentationUncertainty):
#
#    def metric(self, probs: np.array) -> float:
#        """
#        :param probs: array [num_classes, *image_shape]
#        :return: float
#        """
#        p = np.sum(probs[1:], axis=0)  # probability of foreground
#        
#        return -np.sum((p-0.5)**2)
    
#funções de medidas de incertezas
def entropy_of_expected(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_classes]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=0)

def entropy_of_expected_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(entropy_of_expected(probs, epsilon=epsilon))

def expected_entropy(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    log_probs = -np.log(probs + epsilon)
    return np.mean(np.sum(probs * log_probs, axis=1), axis=0)

def expected_entropy_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(expected_entropy(probs, epsilon=epsilon))

def max_prob_uncertainty(probs):
    """
    Maximum probability over classes.

    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: float
    """
    mean_probs = np.mean(probs, axis=0)

    if mean_probs.shape[0] == 1:  # binary classification, explicit background
                                  # probabilities
        background = 1. - mean_probs[0]
        mean_probs = np.stack([background, mean_probs[0]], axis=0)

    confidence = mean_probs.max(0)  # probability of the predicted class

    return -np.mean(confidence)

def max_confidence_uncertainty(probs):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return -np.mean(np.max(probs.reshape((probs.shape[0],-1)), axis=-1))

def max_softmax_uncertainty(probs):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return -np.mean(softmax(probs.reshape((probs.shape[0],-1)), axis=-1).max())

def mutual_information(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    exe = expected_entropy(probs, epsilon=epsilon)
    eoe = entropy_of_expected(probs, epsilon=epsilon)

    return eoe - exe

def mutual_information_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(mutual_information(probs, epsilon=epsilon))

def expected_pw_kl(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)

    exe = expected_entropy(probs, epsilon=epsilon)

    return -np.sum(mean_probs * mean_lprobs, axis=0) - exe

def expected_pw_kl_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(expected_pw_kl(probs, epsilon=epsilon))

def reverse_mutual_information(probs, epsilon=1e-10):
    epkl = expected_pw_kl(probs, epsilon=epsilon)
    mi = mutual_information(probs, epsilon=epsilon)

    return epkl - mi

def reverse_mutual_information_uncertainty(probs, epsilon=1e-10):
    if probs.shape[1] == 1:  # binary classification, explicit background
                             # probabilities
        background = 1. - probs[:,0]
        probs = np.stack([background, probs[:,0]], axis=1)

    return np.mean(reverse_mutual_information(probs, epsilon=epsilon))

def naive_pred_size_uncertainty(probs):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: float
    """
    mean_probs = np.mean(probs, axis=0)
    return -np.sum(mean_probs > 0.5)

def ensemble_uncertainties_classification(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_classes, num_voxels_X, num_voxels_Y, num_voxels_Z]
    :return: Dictionary of uncertainties
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected(probs, epsilon)
    exe = expected_entropy(probs, epsilon)

    mutual_info = eoe - exe

    epkl = -np.sum(mean_probs * mean_lprobs, axis=-1) - exe

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'epkl': epkl,
                   'reverse_mutual_information': epkl - mutual_info,
                   }

    return uncertainty
