import torch
from torch import nn
from unet_modules import UnetWarp
from autoencoder import AutoEncoder
from edm_scheduler import EDMDiffuser, EDMSolver

class DiffusionModel(nn.Module):
    # TODO: implement EMA model
    def __init__(self,
                 net_config,
                 diffusion_config,
                 ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_class = net_config['n_class']
        self.ae = AutoEncoder().to(self.device)
        self.net = UnetWarp(net_config).to(self.device)
        self.diffuser = EDMDiffuser(**diffusion_config).to(self.device)
        self.solver = EDMSolver(**diffusion_config).to(self.device)
    
    def calculate_loss(self, x, sigma, x_pred):
        return self.diffuser.calculate_loss(x, sigma, x_pred)
    
    def train_step(self, x0, cls):
        log_sigma, sigma = self.diffuser.sample_sigma(x0.shape[0], device=self.device)
        x = self.diffuser.diffuse(x0,sigma)
        x_pred = self(x, cls, log_sigma, sigma, cls_mask_ratio=0.15)
        return self.calculate_loss(x0, sigma, x_pred)
    
    @torch.no_grad()
    def condional_generation(self,
                             cls,
                             guidance_scale:int=1,
                             batch_size:int=16,
                             ):
        if isinstance(cls, int):
            cls = torch.full([batch_size], cls, dtype=torch.long).to(self.device)
        x, _, _ = self.solver.solve(self,
                                    cls,
                                    guidance_scale,
                                    batch_size,
                                    n_steps=1024,
                                    n_middle_steps=0)
        return self.decode(x)
    
    @torch.no_grad()
    def guided_eval(self, x, cls, t, guidance=1):
        if guidance>1:
            # Conditional and unconditional outputs
            D_cond = self(x, cls, t, cls_mask_ratio=0.0)      # Conditional denoising
            D_uncond = self(x, cls, t, cls_mask_ratio=1.0)  # Unconditional denoising
            # Linear combination based on guidance scale
            return guidance * D_cond + (guidance - 1) * D_uncond
        else:
            return self(x, cls, t, cls_mask_ratio=0.0)

    @torch.no_grad()
    def condional_generation_with_middle_steps(self,
                                                cls,
                                                guidance_scale:int=1,
                                                batch_size:int=4,
                                                n_middle_steps=8
                                                ):
        if isinstance(cls, int):
            cls = torch.full([batch_size], cls, dtype=torch.long).to(self.device)
        _, x_list, x0_pred_list = self.solver.solve(self,
                                                    cls,
                                                    guidance_scale,
                                                    batch_size,
                                                    n_steps=1024,
                                                    n_middle_steps=n_middle_steps)
        x_list, x0_pred_list = torch.vstack(x_list), torch.vstack(x0_pred_list)
        img = self.decode(x_list)
        img0 = self.decode(x0_pred_list)
        return img, img0
        
        
    @torch.no_grad()
    def decode(self, x, need_postprocess=True):
        if need_postprocess:
            x = self.ae.postprocess(x)
        return self.ae.decode(x)
    
    @torch.no_grad()
    def encode(self, img):
        return self.ae.encode(img)
    
    def forward(self, x, cls, sigma, log_sigma=None, cls_mask_ratio=0.0):
        if cls_mask_ratio > 0:
            mask=torch.rand(x.shape[0]) < cls_mask_ratio
        else:
            mask=None
        cskip, cout, cin, cnoise = self.diffuser.get_c(sigma,log_sigma)
        x_in = x*cin[:,None,None,None]
        x_skip = x*cskip[:,None,None,None]
        x_pred = cout[:,None,None,None] * self.net(x_in, cnoise, cls, mask) + x_skip
        return x_pred
