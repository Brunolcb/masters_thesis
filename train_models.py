from pathlib import Path

# from src.train import train_unet
from src.train import train_le, train_unet


DATA_DIR = Path('data/raw/Dataset_BUSI_with_GT/')

if __name__ == '__main__':
    train_le(DATA_DIR)

    for i in range(5):
        train_unet(DATA_DIR, i)
