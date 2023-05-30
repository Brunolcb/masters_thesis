from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import wandb

from tqdm import tqdm

from .metrics import dice_coef
from .data import get_loaders, get_loaders_test, make_dataset
from .net import UNET, LayerEnsembles, load_checkpoint, save_checkpoint


def check_accuracy(loader, model, loss_fn, epoch, device="cuda", layer='final'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x)[layer])
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += dice_coef(preds, y)
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    
    print(f"Dice score: {dice_score/len(loader)}")
    if layer == 'final':
        wandb.log({ "val_loss_%s"%layer: loss_fn(preds, y), "val_dice_%s"%layer: dice_score/len(loader)}, step=epoch, commit = True)
    else:
        wandb.log({ "val_loss_%s"%layer: loss_fn(preds, y), "val_dice_%s"%layer: dice_score/len(loader)}, step=epoch, commit = False) 
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()

def inference(hyperparameters, model, X, y):

    test_loader = get_loaders_test(X,y)
    load_checkpoint(torch.load("models/model_LE.pth.tar", map_location=hyperparameters['device']), model)
    device=hyperparameters['device']
    tests = []
    tests_pred = []
    tests_pred_bin = []
    x_tests = []
    count_test = 0
    total_dice = 0
    
    model.eval()
    for idx, (x, y) in enumerate(test_loader):
        dice_score = 0
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            outputs_tests = model(x)
            test_preds = [torch.sigmoid(out) for _, out in outputs_tests.items()]
            test_preds_bin = [(torch.sigmoid(out)> 0.5).float() for _, out in outputs_tests.items()]
        tests.append(y)
        tests_pred.append(test_preds)
        tests_pred_bin.append(test_preds_bin)
        x_tests.append(x)
        dice_score += dice_coef(test_preds_bin[-1], y)
        total_dice += dice_score
        #wandb.log({"dice_per_test": dice_score}, step=count_test)
        count_test +=1
        
    return x_tests, tests, tests_pred, tests_pred_bin

def data_pass(model, loader, loss_fn, optimizer=None, scaler=None,
              device=torch.device('cpu')):
    epoch_dice = 0
    epoch_loss = 0
    for data, targets in loader:
        data = data.to(device)
        targets = targets.float().to(device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            if isinstance(predictions, dict):  # LayerEnsemble
                loss = sum([loss_fn(pred, targets) for pred in predictions.values()]) / len(predictions)
                predictions = predictions['final']
            else:
                loss = loss_fn(predictions, targets)
    
        # backward
        if optimizer is not None:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # metrics
        train_preds = torch.sigmoid(predictions)
        train_preds = (train_preds > 0.5).float()
        epoch_dice += dice_coef(train_preds, targets).item()
        epoch_loss += loss.item() * len(targets)
    epoch_loss = epoch_loss / len(loader.dataset)
    epoch_dice = epoch_dice / len(loader)

    return epoch_loss, epoch_dice

def _train(model, model_fname, data_dir, device='cuda', lr=5e-5, batch_size=32,
           num_epochs=10, num_workers=2, pin_memory=True, model_dir='models/'):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    device = torch.device(device)

    # load data
    X_train, y_train, X_val, y_val, _, _ = make_dataset(data_dir)

    train_loader, val_loader = get_loaders(
        X_train, y_train, X_val, y_val,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # load training stuff
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = list()
    val_losses = list()
    val_dices = list()
    for epoch in tqdm(list(range(num_epochs))):
        model.train()
        train_loss, _ = data_pass(model, train_loader, loss_fn, optimizer,
                                  scaler, device)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss, val_dice = data_pass(model, val_loader, loss_fn,
                                           device=device)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=model_dir/model_fname)
    
    return model, train_losses, val_losses, val_dices

def train_unet(data_dir: Path, model_num=0, device='cuda', lr=5e-5, batch_size=32, num_epochs=10, num_workers=2, pin_memory=True, model_dir='models/'):
    # load model
    model = UNET(in_channels=1, out_channels=1).to(device)

    model_fname = "unet_%s.pth.tar" % model_num

    return _train(model, model_fname, data_dir, device, lr, batch_size, num_epochs, num_workers, pin_memory, model_dir)


def train_le(data_dir: Path, device='cuda', lr=5e-5, batch_size=32, num_epochs=10, num_workers=2, pin_memory=True, model_dir='models/'):
    # load model
    model = LayerEnsembles.from_UNET(UNET(in_channels=1, out_channels=1).to(device))

    model_fname = "le.pth.tar"

    return _train(model, model_fname, data_dir, device, lr, batch_size, num_epochs, num_workers, pin_memory, model_dir)

def predict(loader, model):
    Y_hat = list()
    Y = list()
    for imgs, targets in loader:
        with torch.no_grad():
            y_hats = model(imgs)

        if isinstance(y_hats, dict):
            y_hats = list(y_hats.values())
            y_hats = torch.stack(y_hats)

        Y_hat.append(y_hats)
        Y.append(targets)
    Y_hat = torch.cat(Y_hat, dim=1)
    Y = torch.vstack(Y)

    return Y_hat, Y
