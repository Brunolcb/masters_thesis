import torch
import torchvision
import wandb

from .metrics import dice_coef
from .data import get_loaders_test
from .net import load_checkpoint


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