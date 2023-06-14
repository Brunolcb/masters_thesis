from pathlib import Path

import numpy as np
import torch

from src.data import get_loaders, make_dataset
from src.net import LayerEnsembles, UNET, UNETEnsemble
from src.train import predict


DATA_DIR = Path('data/raw/Dataset_BUSI_with_GT/')

if __name__ == '__main__':
    _, _, X_val, y_val, X_test, y_test = make_dataset(DATA_DIR)

    val_loader, test_loader = get_loaders(X_val,y_val,X_test,y_test,16,shuffle=False)

    models_dir = Path('models/')
    le = LayerEnsembles.from_UNET(UNET(in_channels=1, out_channels=1))
    le.load_state_dict(torch.load(models_dir/'model_LE.pth.tar')['state_dict'], le)

    unets = list()
    for i in range(5):
        model_fpath = models_dir/f"model_{i}.pth.tar"
        unet = UNET(in_channels=1, out_channels=1)
        unet.load_state_dict(torch.load(model_fpath)['state_dict'], unet)
        unets.append(unet)
    de = UNETEnsemble(unets)

    # predict and save
    Y_hat, Y = predict(val_loader, de)
    np.savez_compressed(models_dir/'DE_validation.npz',
                        y_hat=Y_hat.numpy(), y=Y.numpy())

    Y_hat, Y = predict(val_loader, le)
    np.savez_compressed(models_dir/'LE_validation.npz',
                        y_hat=Y_hat.numpy(), y=Y.numpy())

    Y_hat, Y = predict(test_loader, de)
    np.savez_compressed(models_dir/'DE_test.npz',
                        y_hat=Y_hat.numpy(), y=Y.numpy())

    Y_hat, Y = predict(test_loader, le)
    np.savez_compressed(models_dir/'LE_test.npz',
                        y_hat=Y_hat.numpy(), y=Y.numpy())
