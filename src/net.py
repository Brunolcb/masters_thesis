import torch
import torchvision.transforms.functional as TF

from torch import nn, Tensor
from typing import Dict, Callable, Iterable


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[32,64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        
        self.downs = nn.ModuleList()
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        #self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=None, activation=None, upsampling=1):
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        #activation = nn.Sigmoid()
        activation = nn.Identity()
        super().__init__(dropout, conv2d, upsampling, activation)

class LayerEnsembles(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
        self._layer_ensemble_active = False

    def set_output_heads(self, in_channels: Iterable[int],
                         scale_factors: Iterable[int], classes: int,
                         pooling: str = 'avg', dropout: float = None,
                         device ='cpu'):
        if self._layer_ensemble_active:
            raise ValueError("Output heads should be set only once.")
        self._layer_ensemble_active = True
        
        self.output_heads = nn.ModuleList([
            SegmentationHead(
                in_channels=in_channel,
                out_channels=classes,
                kernel_size=3,
                dropout=dropout,
                upsampling=scale_factor,
            ) for in_channel, scale_factor in zip(in_channels, scale_factors)
        ]).to(device)

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        final_layer_output = self.model(x)
        if not self._layer_ensemble_active:
            outputs = {layer: self._features[layer] for layer in self.layers}
        else: 
            outputs = {layer: head(self._features[layer]) for head, layer in zip(self.output_heads, self.layers)}
        outputs['final'] = final_layer_output
        return outputs
