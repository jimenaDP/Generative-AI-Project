import torch
import numpy as np
import diffusion_process as dfp

class DiffusionModel:
    def __init__(self, type, sigma):
        self.type = type
        self.sigma = sigma

    def drift(self, x_t, t):
        if self.type == "VE":
            return torch.zeros_like(x_t) #drift = 0
        else:
            raise NotImplementedError
        
    def diffusion(self, t):
        if self.type == "VE":
            return torch.pow(self.sigma, t)  #g(t) = σ^t
        else:
            raise NotImplementedError
        
    def mu_t(self, x_0, t):
        if self.type == "VE":
            return x_0
        else:
            raise NotImplementedError
    
    def sigma_t(self, t):
        if self.type == "VE":
            return torch.sqrt(
                0.5 * (self.sigma ** (2 * t) - 1.0) / np.log(self.sigma)
            )
        else:
            raise NotImplementedError
        
    def build(self):
        return dfp.GaussianDiffussionProcess(
            drift_coefficient=self.drift,
            diffusion_coefficient=self.diffusion,
            mu_t=self.mu_t,
            sigma_t=self.sigma_t,
        )

    