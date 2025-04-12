import torch

class IdealPosteriorEstimatorEDM:
    def __init__(self, clean_data, device):
        self.device=device
        self.clean_data = clean_data  # shape: [num_samples, ...]
        self.num_samples = clean_data.shape[0]
        self.data_shape = clean_data.shape[1:]

    def __call__(self, xt, t):
        if isinstance(t, (int, float)):
            t = torch.full([xt.shape[0]], t).to(self.device)
        elif isinstance(t, torch.Tensor) and t.dim()==0:
            t=t.expand(xt.shape[0]).to(self.device)
        return torch.stack([self._compute_single_posterior(x, ti) for x, ti in zip(xt, t)])

    def _compute_single_posterior(self, xt, t):
        t_sq = t**2
        exponents = -0.5 * torch.sum((xt - self.clean_data)**2, 
                      dim=(1,2,3)) / t_sq
        weights = torch.softmax(exponents, dim=0)
        return torch.sum(self.clean_data * weights.view(-1, 1, 1, 1), dim=0)

    def guided_eval(self, xt, cls, t, guidance, *args, **kwargs):
        return self(xt, t)
    

class IdealPosteriorEstimatorFM(IdealPosteriorEstimatorEDM):
    def __call__(self, xt, t):
        if isinstance(t, (int, float)):
            t = torch.full([xt.shape[0]], t).to(self.device)
        elif isinstance(t, torch.Tensor) and t.dim()==0:
            t=t.expand(xt.shape[0]).to(self.device)
        return torch.stack([self._compute_single_posterior(x, ti) for x, ti in zip(xt, t)])

    def _compute_single_posterior(self, xt, t):
        t_sq = t**2
        exponents = -0.5 * torch.sum((xt - self.clean_data)**2, 
                      dim=(1,2,3)) / t_sq
        weights = torch.softmax(exponents, dim=0)
        return (xt-torch.sum(self.clean_data * weights.view(-1, 1, 1, 1), dim=0))/t

    def guided_eval(self, xt, cls, t, guidance, *args, **kwargs):
        return self(xt, t)
