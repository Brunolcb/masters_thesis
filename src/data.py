import os
from pathlib import Path
import torch
import numpy as np

from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def make_dataset(path_to_dir: Path):
    X, y = get_Xy(path_to_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=1)

    # augmentation performed on the training set
    aug_X_train = list()
    aug_y_train = list()
    for image, mask in zip(X_train, y_train):
        aug_X_train.append(image)
        for aug_image in get_augmentations(image):
            aug_X_train.append(aug_image)

        aug_y_train.append(mask)
        for aug_mask in get_augmentations(mask):
            aug_y_train.append(aug_mask)
    X_train = np.array(aug_X_train)
    y_train = np.array(aug_y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_augmentations(img):
    horizontal_flip = np.flip(img, 1)
    vertical_horizontal_flip = np.flip(img, (0,1))
    imagerotate = np.rot90(img, 2) 
    imagerotate1 = np.rot90(img, 1) 
    imagerotate2 = np.rot90(img, 3) 

    return horizontal_flip, vertical_horizontal_flip, imagerotate, imagerotate1, imagerotate2

def get_Xy(path_to_dir: Path):
    path_to_dir = Path(path_to_dir)

    #x for images 
    #y for masks
    #t for target"label"
    X_b, y_b = np.zeros((437, 128, 128, 1)), np.zeros((437, 128, 128, 1))
    X_n, y_n = np.zeros((133, 128, 128, 1)), np.zeros((133, 128, 128, 1))
    X_m, y_m= np.zeros((210, 128, 128, 1)), np.zeros((210, 128, 128, 1))

    for img, tumor_type in enumerate(os.listdir(path_to_dir)) :
        for image in os.listdir(path_to_dir/tumor_type) :
            p = os.path.join(path_to_dir/tumor_type, image)
            # img = cv2.imread(p,cv2.IMREAD_GRAYSCALE)           # read image as  grayscale
            pil_img = Image.open(p).convert("L")
        
            if '_mask' in image:
                # img = cv2.resize(img,(128,128))
                # pil_img = Image.fromarray(img)
                # pil_img = pil_img.resize((128, 128))
                pil_img = pil_img.resize((128, 128), Image.NEAREST)
                
                # some images have multiple separate masks for multiple tumors 
                if image[0] == 'b' :
                    y_b[get_image_num(image)-1] += np.array(pil_img).reshape(128,128,1)    
                if image[0] == 'n' :
                    y_n[get_image_num(image)-1] += np.array(pil_img).reshape(128,128,1)  
                if image[0] == 'm' :
                    y_m[get_image_num(image)-1] += np.array(pil_img).reshape(128,128,1)  
            else:
                # img = cv2.resize(img,(128,128))

                # pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((128, 128))

                if image[0] == 'b' :
                    X_b[get_image_num(image)-1] = np.array(pil_img).reshape(128,128,1)  
                if image[0] == 'n' :
                    X_n[get_image_num(image)-1] = np.array(pil_img).reshape(128,128,1)  
                if image[0] == 'm' :
                    X_m[get_image_num(image)-1] = np.array(pil_img).reshape(128,128,1)  
    X = np.concatenate((X_b, X_n, X_m), axis = 0)
    y = np.concatenate((y_b, y_n, y_m), axis = 0)

    X = X / 255
    y = y / 255

    y[y > 1.] = 1.

    assert X.max() <= 1.

    return X, y

def get_image_num(image):
    val = 0
    
    for i in range(len(image)) :
        if image[i] == '(' :
            while True :
                i += 1
                if image[i] == ')' :
                    break
                val = (val*10) + int(image[i])
            break
    
    return val

def get_loaders(X_train,y_train,X_val,y_val,batch_size,num_workers=4,pin_memory=True,shuffle=True):
    X_train_tensor = torch.Tensor(X_train.reshape(-1,1,128,128))
    y_train_tensor = torch.Tensor(y_train.reshape(-1,1,128,128))
    
    train_loader = DataLoader(
        TensorDataset(X_train_tensor,y_train_tensor),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    ) 

    X_val_tensor = torch.Tensor(X_val.reshape(-1,1,128,128))
    y_val_tensor = torch.Tensor(y_val.reshape(-1,1,128,128))
    
    val_loader = DataLoader(
        TensorDataset(X_val_tensor,y_val_tensor),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    ) 

    return train_loader, val_loader    

def get_loaders_test(X_test,y_test):
    
    X_test_tensor = torch.Tensor(X_test.reshape(-1,1,128,128))
    y_test_tensor = torch.Tensor(y_test.reshape(-1,1,128,128))
    
    test_loader = DataLoader(TensorDataset(X_test_tensor,y_test_tensor)) 

    return test_loader   