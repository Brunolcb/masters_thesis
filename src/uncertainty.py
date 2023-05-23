import numpy as np


#incertezas calculadas a ser selecionada a partir da ultima até a apontada
def uncertainties_heads(preds,heads):
    uncertainty_cat = []
    for i in range(len(preds[0])-1,heads-1,-1):
        choosen_heads = [pred[i].cpu().numpy().reshape(128,128,1) for pred in preds]
        uncertainty_cat.append(choosen_heads)
        
    uncertainty_cat = np.array(uncertainty_cat,dtype='float32')
    
    return np.concatenate((uncertainty_cat, 1-uncertainty_cat), axis = -1)

#incertezas da ultima cabeça
def uncertainties_final_heads(preds):
    uncertainty_cat =[pred[-1].cpu().numpy().reshape(128,128,1) for pred in preds]
    uncertainty_cat = np.array([uncertainty_cat],dtype='float32')
    
    return np.concatenate((uncertainty_cat, 1-uncertainty_cat), axis = -1)

#funções de medidas de incertezas
def entropy_of_expected(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_voxels_X, num_voxels_Y, num_classes]
    :return: array [num_voxels_X, num_voxels_Y]
    """
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=-1)

def expected_entropy(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
    :return: array [num_voxels_X, num_voxels_Y, num_voxels_Z,]
    """
    log_probs = -np.log(probs + epsilon)
    return np.mean(np.sum(probs * log_probs, axis=-1), axis=0)


def ensemble_uncertainties_classification(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_voxels_X, num_voxels_Y, num_voxels_Z, num_classes]
    :return: Dictionary of uncertainties
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)
    conf = np.max(mean_probs, axis=-1)

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