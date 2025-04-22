import torch
from unet_modules import UnetWrap
from autoencoder import AutoEncoder
from diffusion_model_edm_unet import DiffusionModel as EDM
from fm_scheduler import FMDiffuser,FMSolver
from ideal_net import IdealPosteriorEstimatorFM

class DiffusionModel(EDM):
    # TODO: EMA
    # TODO: optimal data-noise pairing
    def build_modules(self,net_config,diffusion_config):
        self.ae = AutoEncoder(sigma_data=1.0).to(self.device)
        self.net = UnetWrap(net_config).to(self.device)
        self.diffuser = FMDiffuser(self.device, **diffusion_config).to(self.device)
        self.solver = FMSolver(self.device, **diffusion_config).to(self.device)
    
    def calculate_loss(self, v, v_pred):
        return (v-v_pred).pow(2).mean()
    
    def train_step(self, x0, cls):
        t = self.diffuser.sample_t(x0.shape[0], device=self.device)
        x, v = self.diffuser.diffuse(x0,t)
        v_pred = self(x, cls, t, cls_mask_ratio=0.15)
        return self.calculate_loss(v, v_pred)
    
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
        x, _ = self.diffuser.diffuse(x0, t)
        v_pred = self(x, cls, t, cls_mask_ratio=0.0)
        x_pred = x - v_pred * t[:,None,None,None]
        return self.decode(x_pred)
    
    @torch.no_grad()
    def eval_solver(self, x, batch_size=9):
        ideal_net = IdealPosteriorEstimatorFM(x, self.device)
        x, _, _ = self.solver.solve(ideal_net,
                                    cls=0,
                                    guidance_scale=0,
                                    batch_size=batch_size,
                                    use_2nd_order=True,
                                    n_steps=8,
                                    n_middle_steps=0)
        return self.decode(x, need_postprocess=False)
    
    # @torch.compile
    def forward(self, x, cls, t, cls_mask_ratio=0.0):
        if cls_mask_ratio > 0:
            mask=torch.rand(x.shape[0]) < cls_mask_ratio
        else:
            mask=None
        return self.net(x, t, cls, mask)

    
