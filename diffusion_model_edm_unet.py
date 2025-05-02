import torch
from torch import nn
from unet_modules import UnetWrap
from autoencoder import AutoEncoder
from edm_scheduler import EDMDiffuser, EDMSolver
from ideal_net import IdealPosteriorEstimatorEDM

class DiffusionModel(nn.Module):
    # TODO: implement EMA model
    def __init__(self,
                 net_config,
                 diffusion_config,
                 ):
        super().__init__()
        self.net_config = net_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_class = net_config['n_class']
        self.build_modules(net_config, diffusion_config)
    
    def build_modules(self,net_config,diffusion_config):
        self.ae = AutoEncoder(sigma_data=0.5).to(self.device)
        self.net = UnetWrap(net_config).to(self.device)
        self.diffuser = EDMDiffuser(self.device, **diffusion_config).to(self.device)
        self.solver = EDMSolver(self.device, **diffusion_config).to(self.device)
    
    def init_ema(self):
        assert not hasattr(self, 'ema_net'), "ema_net already exists!"
        # copy self.net as init
        self.ema_net=UnetWrap(self.net_config).to(self.device)
        self.ema_net.load_state_dict(self.net.state_dict())
    
    @torch.no_grad()
    def ema_update(self, decay=0.999):
        if hasattr(self, 'ema_net'):
            # update ema_net with decay
            for ema_param, param in zip(self.ema_net.parameters(), self.net.parameters()):
                ema_param.data = decay * ema_param.data + (1 - decay) * param.data
    
    @torch.no_grad()
    def apply_ema(self):
        if hasattr(self, 'ema_net'):
            # apply ema_net to net
            self.net.load_state_dict(self.ema_net.state_dict())
            # del self.ema_net
    
    def calculate_loss(self, x, sigma, x_pred):
        return self.diffuser.calculate_loss(x, sigma, x_pred)
    
    #@torch.compile(backend='cudagraphs')
    def train_step(self, x0, cls):
        sigma, log_sigma = self.diffuser.sample_sigma(x0.shape[0], device=self.device)
        x = self.diffuser.diffuse(x0,sigma)
        x_pred = self(x, cls, sigma, log_sigma, cls_mask_ratio=0.15)
        return self.calculate_loss(x0, sigma, x_pred)
    
    @torch.no_grad()
    def guided_eval(self, x, cls, t, guidance=1, cfg_scale=False):
        if isinstance(cls, int):
            cls = torch.full([x.shape[0]], cls, dtype=torch.long).to(self.device)
        if isinstance(t, (int,float)):
            t = torch.full([x.shape[0]], t).to(self.device)
        elif isinstance(t, torch.Tensor) and t.dim()==0:
            t = t.expand(x.shape[0]).to(self.device)
        if guidance>1:
            # Conditional and unconditional outputs
            D_cond = self(x, cls, t, cls_mask_ratio=0.0)      # Conditional denoising
            D_uncond = self(x, cls, t, cls_mask_ratio=1.0)  # Unconditional denoising
            # Linear combination based on guidance scale
            if cfg_scale:
                # only works for velocity prediction
                s=(D_cond*D_uncond).sum(dim=(1,2,3))/\
                    (D_uncond.pow(2).sum(dim=(1,2,3))+1e-8)
                s=s[:,None,None,None]
            else:
                s=1
            return guidance * D_cond - (guidance - 1) * s * D_uncond
        else:
            return self(x, cls, t, cls_mask_ratio=0.0)

    @torch.no_grad()
    def conditional_generation(self,
                             cls,
                             guidance_scale:int=1,
                             batch_size:int=16,
                             use_2nd_order=False,
                             n_steps=512,
                             cfg_zero_star=False
                             ):
        x, _, _ = self.solver.solve(self,
                                    cls,
                                    guidance_scale,
                                    batch_size,
                                    use_2nd_order=use_2nd_order,
                                    n_steps=n_steps,
                                    n_middle_steps=0,
                                    cfg_zero_star=cfg_zero_star
                                    )
        return self.decode(x)
    
    @torch.no_grad()
    def conditional_generation_single_step(self,
                                            x0,
                                            cls,
                                            t,
                                            batch_size:int=8,
                                            ):
        x0 = x0[:batch_size]
        if isinstance(t, (int,float)):
            t = torch.full([x0.shape[0]], t).to(self.device)
        elif isinstance(t, torch.Tensor) and t.dim()==0:
            t = t.expand(x0.shape[0]).to(self.device)
        x = self.diffuser.diffuse(x0, t)
        x_pred = self(x, cls, t, torch.log(t), cls_mask_ratio=0.0)
        return self.decode(x_pred)

    @torch.no_grad()
    def conditional_generation_with_middle_steps(self,
                                                 cls,
                                                 guidance_scale:int=1,
                                                 use_2nd_order=False,
                                                 batch_size:int=4,
                                                 n_steps=512,
                                                 n_middle_steps=8
                                                 ):
        
        _, x_list, x0_pred_list = self.solver.solve(self,
                                                    cls,
                                                    guidance_scale,
                                                    batch_size,
                                                    use_2nd_order=use_2nd_order,
                                                    n_steps=n_steps,
                                                    n_middle_steps=n_middle_steps)
        x_list, x0_pred_list = torch.vstack(x_list), torch.vstack(x0_pred_list)
        img = self.decode(x_list)
        img0 = self.decode(x0_pred_list)
        return img, img0
    
    @torch.no_grad()
    def eval_solver(self, x, batch_size=9):
        ideal_net = IdealPosteriorEstimatorEDM(x, self.device)
        x, _, _ = self.solver.solve(ideal_net,
                                    cls=0,
                                    guidance_scale=0,
                                    batch_size=batch_size,
                                    use_2nd_order=True,
                                    n_steps=64,
                                    n_middle_steps=0)
        return self.decode(x, need_postprocess=False)
        
    @torch.no_grad()
    def decode(self, x, need_postprocess=True):
        if need_postprocess:
            x = self.ae.postprocess(x)
        return self.ae.decode(x)
    
    @torch.no_grad()
    def encode(self, img):
        return self.ae.encode(img)
    
    # @torch.compile
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
