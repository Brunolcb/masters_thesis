import os

import numpy as np
import dask.array as da

from PIL import Image

from dask.distributed import Client, progress


if __name__ == '__main__':
    client = Client()

    path = '/data/busi_wo_null_train_masks'

    masks = [np.array(Image.open(os.path.join(path, file)).resize((256,256))).astype('float32') for file in os.listdir(path)]
    masks = np.stack(masks)

    Y = masks.reshape(masks.shape[0],-1)
    Y = da.from_array(Y, chunks=(Y.shape[0],512))

    cov = da.cov(Y.T)

    # add small values to diagonal to avoid non positive-definiteness due to
    # numerical errors
    L = da.linalg.cholesky(cov + da.eye(cov.shape[0]) * 1e-3, lower=True)

    da.to_npy_stack('./data/interim/BUSI_L_cov/', L)
