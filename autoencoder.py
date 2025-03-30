from diffusers.models import AutoencoderKL
from torch import nn
import torch


class AutoEncoder(nn.Module):

    def __init__(self,
                 model_type="stabilityai/sd-vae-ft-ema",
                 # params = 34 + 49 M
                 # f = 8; embed_dim = 4
                 **ignoredkeys):

        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type).eval().requires_grad_(False)
        self.sigma_data = 0.5
        # remember to scale mean here when changing resolution
        self.m = nn.Parameter(torch.tensor([5.81, 3.25, 0.12, -2.15]).view(4,1,1)/4, 
                                requires_grad=False)
        self.s = nn.Parameter(torch.tensor([4.17, 4.62, 3.71, 3.28]).view(4,1,1), 
                                requires_grad=False)
        self.precomputed_val = - self.m / self.s * self.sigma_data


    def forward(self, x):
        return self.model(x).sample

    @torch.compile
    def encode(self, x, mode=True):
        dist = self.model.encode(x).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()

    @torch.compile
    def decode(self, x):
        return self.model.decode(x).sample

    def preprocess(self, x):
        # (x-m)/s*sigma_data
        return torch.addcdiv(self.precomputed_val, x, self.s, value=self.sigma_data)
        
    def postprocess(self, x):
        # x*s/sigma_data + m
        return torch.addcmul(self.m, x, self.s, value=1/self.sigma_data)