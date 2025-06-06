import torch
from torch import nn

class EDMDiffuser(nn.Module):
    def __init__(self, device,**ignoredkwargs):
        super().__init__()
        self.device=device
        self.sigma_data = 0.5
        self.Pm, self.Ps = -0.4, 1.0  # depends on sigma data
    
    @torch.no_grad()
    def get_c(self, sigma: torch.Tensor, log_sigma: torch.Tensor = None):
        sigma_data = self.sigma_data
        skip = sigma_data**2 / (sigma.pow(2)+sigma_data**2)
        in_ = 1 / (sigma.pow(2)+sigma_data**2).sqrt()
        out = (sigma*sigma_data) * in_
        noise = log_sigma/4 if log_sigma is not None else torch.log(sigma)/4
        return skip,out,in_,noise
    
    @torch.no_grad()
    def sample_sigma(self, batch_size:int, device=torch.device('cuda')):
        log_sigma = (torch.randn(batch_size) * self.Ps + self.Pm).to(device)
        return torch.exp(log_sigma), log_sigma
    
    @torch.no_grad()
    def diffuse(self, x, sigma, require_n=False):
        # VE
        n = torch.randn_like(x) * sigma[:, None, None, None]
        if require_n:
            return x+n, n
        return x + n

    def calculate_loss(self, x, sigma, x_pred):
        # need revisiting for numeric stability
        lambda_ = (sigma.pow(2) + self.sigma_data ** 2) /\
                    (sigma*self.sigma_data).pow(2)
        return (torch.mean((x-x_pred).pow(2),dim=(1,2,3)) * lambda_).mean()


class EDMSolver(nn.Module):
    def __init__(self,device, **ignoredkwargs):
        super().__init__()
        self.device=device
        self.rho = 7
        self.sigma_min = 0.002 # depends on sigma data
        self.sigma_max = 80  # depends on sigma data
    
    @torch.no_grad()
    def solve(self, *args, **kwargs):
        return self.solve_sde(*args, **kwargs)

    @torch.no_grad()
    def solve_sde(self,                 
                  model,
                  cls,
                  guidance_scale,
                  batch_size,
                  use_2nd_order=True,
                  n_steps=1024,
                  n_middle_steps=0,
                  S=(40,0.05,50,1.003), # Schurn, Stmin, Stmax, Snoise
                  cfg_zero_star=False,
                  **ignoredkwargs):
        if isinstance(cfg_zero_star,bool):
            cfg_zero_star=(cfg_zero_star,cfg_zero_star) # zero_init, rescale
        if S is None: # deterministic
            S=(0,0,0,1.0)
        # a variant of the Euler-Maruyama method from EDM
        t = torch.arange(0,n_steps+1)
        #self.rho=10-8*(t/t.max())
        t = (self.sigma_max**(1/self.rho)+\
             t/n_steps*\
             (self.sigma_min**(1/self.rho)-\
             self.sigma_max**(1/self.rho)))**self.rho
        gamma = [min(2**0.5-1,S[0]/n_steps) if S[1]<t[i]<S[2] else 0\
                 for i in range(n_steps)]
        t = t.to(self.device)
        x_pred_list = []
        log_every_n_steps = n_steps // (1 + n_middle_steps)
        xi = torch.randn(batch_size, 4, 32, 32).to(model.device) * t[0]
        for i in range(n_steps):
            if i < 2 and cfg_zero_star[0]:
                continue
            epsilon = torch.randn_like(xi) * S[3]
            ti_hat = (1+gamma[i])*t[i]
            xi_hat = xi + t[i] * epsilon * (2*gamma[i]+gamma[i]**2)**0.5
            di = (xi_hat - model.guided_eval(xi_hat,
                                             cls,
                                             ti_hat,
                                             guidance_scale,
                                             cfg_scale=cfg_zero_star[1])) / ti_hat
            if model.train_cls_mask_ratio<=0:
                xip1 = xi_hat + (t[i+1] - ti_hat) * di * guidance_scale
            else:
                xip1 = xi_hat + (t[i+1] - ti_hat) * di
            
            if use_2nd_order and t[i+1] > self.sigma_min * 2:
                di_prime = (xip1 - model.guided_eval(xip1,
                                                     cls,
                                                     t[i+1],
                                                     guidance_scale,
                                                     cfg_scale=cfg_zero_star[1]
                                                     )) / t[i+1]
                if model.train_cls_mask_ratio<=0:
                    xip1 = xi_hat + (t[i+1] - ti_hat) * 0.5 * (di + di_prime) * guidance_scale
                else:
                    xip1 = xi_hat + (t[i+1] - ti_hat) * 0.5 * (di + di_prime)
            xi = xip1
            
            if (i+1) % log_every_n_steps == 0:
                x_pred_list.append(xi_hat-ti_hat*di)
        return xi, x_pred_list
    


        
        
        
