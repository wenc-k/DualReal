import torch
import open_clip
import numpy as np
import torch.nn as nn
import torchvision.transforms as T

class FrozenOpenCLIPCustomEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    def __init__(self, pretrained, vit_resolution=(224, 224), arch="ViT-H/14", device="cuda",
                 freeze=True):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained) #适用绝对路径
        self.model = model
        data_white = np.ones((vit_resolution[0], vit_resolution[1], 3), dtype=np.uint8)*255
        self.white_image = preprocess(T.ToPILImage()(data_white)).unsqueeze(0)

        self.device = device

        if freeze:
            self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, image):
        xi = self.model.encode_image(image.to(self.device))
        return xi

    def encode(self, text):
        return self(text)
