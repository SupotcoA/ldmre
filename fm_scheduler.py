import torch
from torch import nn

class FMDiffuser(nn.Module):
    def __init__(self, device, **ignoredkwargs):
        super().__init__()
        self.device=device
        self.sigma_data = 1.0
        self.Pm, self.Ps = 0.0, 1.0  # depends on sigma data
    
    @torch.no_grad()
    def sample_t(self, batch_size:int, device=torch.device('cuda')):
        # logit normal sampling SD3
        u = torch.randn(batch_size).to(device) * self.Ps + self.Pm
        return torch.sigmoid(u)
    
    @torch.no_grad()
    def diffuse(self, x, t):
        # FM
        t_=t[:, None, None, None]
        n = torch.randn_like(x)
        return (1 - t_) * x + t_ * n, n-x # zt, v


class FMSolver(nn.Module):
    # ODE solver: 2nd order Heun
    def __init__(self,device, **ignoredkwargs):
        super().__init__()
        self.device=device
        self.sigma_min = 0.004 # depends on sigma data
    
    @torch.no_grad()
    def solve(self, *args, **kwargs):
        return self.solve_ode(*args, **kwargs)

    @torch.no_grad()
    def solve_ode(self,                 
                  model,
                  cls,
                  guidance_scale,
                  batch_size,
                  use_2nd_order=True,
                  n_steps=1024,
                  n_middle_steps=0,
                  cfg_zero_star=False,
                  **ignoredkwargs):
        if isinstance(cfg_zero_star,bool):
            cfg_zero_star=(cfg_zero_star,cfg_zero_star) # zero_init, rescale
        t = torch.arange(0,n_steps+1)
        t = (1+t/n_steps*(self.sigma_min**(1/self.rho)-1))**self.rho  # EDM
        t = t.to(self.device)
        x_list = []
        x_pred_list = []
        log_every_n_steps = n_steps // (1 + n_middle_steps)
        xi = torch.randn(batch_size, 4, 32, 32).to(model.device) * t[0]
        for i in range(n_steps):
            if i < 2 and cfg_zero_star[0]:
                continue
            di = t[i]*model.guided_eval(xi,
                                        cls,
                                        t[i],
                                        guidance_scale,
                                        cfg_scale=cfg_zero_star[1])
            xip1 = xi + (t[i+1] - t[i]) * di
            
            if use_2nd_order and t[i+1] > self.sigma_min * 2:
                di_prime = t[i+1]* model.guided_eval(xip1,
                                                     cls,
                                                     t[i+1],
                                                     guidance_scale,
                                                     cfg_scale=cfg_zero_star[1]
                                                     )
                xip1 = xi + (t[i+1] - t[i]) * 0.5 * (di + di_prime)
            xi = xip1
            
            if (i+1) % log_every_n_steps == 0:
                x_list.append(xi)
                x_pred_list.append(xi-t[i]*di)
        return xi, x_list, x_pred_list
    


        
        
        
