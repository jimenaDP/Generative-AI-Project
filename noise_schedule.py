import torch


class NoiseSchedule:
    def __init__(
        self,
        schedule,
        sigma_min=None,
        sigma_max=None,
        beta_min=None,
        beta_max=None,
        s=None
    ):
        self.schedule = schedule
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.s = s

    def _s_tensor(self, t):
        return torch.tensor(self.s, device=t.device, dtype=t.dtype)

    # -----------------
    # VE schedules
    # -----------------

    def sigma_t_ve(self, t):
        if self.schedule == "linear":
            return self.sigma_min + t * (self.sigma_max - self.sigma_min)

        elif self.schedule == "cosine":
            s_tensor = self._s_tensor(t)

            alpha_bar = (
                torch.cos((t + s_tensor) / (1 + s_tensor) * torch.pi / 2) ** 2
                / torch.cos((s_tensor / (1 + s_tensor)) * torch.pi / 2) ** 2
            )

            sigma2 = 1.0 - alpha_bar
            sigma2 = torch.clamp(sigma2, min=1.0e-12)
            return torch.sqrt(sigma2)

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VE."
            )

    def diffusion_t_ve(self, t):
        if self.schedule == "linear":
            sigma = self.sigma_t_ve(t)
            return torch.sqrt(2.0 * sigma * (self.sigma_max - self.sigma_min))

        elif self.schedule == "cosine":
            t_req = t.clone().detach().requires_grad_(True)

            sigma = self.sigma_t_ve(t_req)
            sigma2 = sigma ** 2

            grad = torch.autograd.grad(
                outputs=sigma2.sum(),
                inputs=t_req,
                create_graph=False
            )[0]

            g2 = torch.clamp(grad, min=1.0e-12)
            return torch.sqrt(g2)

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VE."
            )

    # -----------------
    # VP schedules
    # -----------------

    def alpha_bar_vp_cosine(self, t):
        s_tensor = self._s_tensor(t)

        a_bar = (
            torch.cos((t + s_tensor) / (1 + s_tensor) * torch.pi / 2) ** 2
            / torch.cos((s_tensor / (1 + s_tensor)) * torch.pi / 2) ** 2
        )

        return torch.clamp(a_bar, min=1.0e-12, max=1.0)

    def beta_t_vp(self, t):
        if self.schedule == "linear":
            return self.beta_min + t * (self.beta_max - self.beta_min)

        elif self.schedule == "cosine":
            beta = (torch.pi / (1 + self.s)) * torch.tan(
                (t + self.s) / (1 + self.s) * torch.pi / 2
            )
            return torch.clamp(beta, min=1.0e-12, max=50.0)

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VP."
            )

    def mu_t_vp(self, x_0, t):
        if self.schedule == "linear":
            integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            coeff = torch.exp(-0.5 * integral)
            return coeff[:, None, None, None] * x_0

        elif self.schedule == "cosine":
            a_bar = self.alpha_bar_vp_cosine(t)
            return torch.sqrt(a_bar)[:, None, None, None] * x_0

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VP."
            )

    def sigma_t_vp(self, t):
        if self.schedule == "linear":
            integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
            return torch.sqrt(1 - torch.exp(-integral))

        elif self.schedule == "cosine":
            a_bar = self.alpha_bar_vp_cosine(t)
            return torch.sqrt(torch.clamp(1.0 - a_bar, min=1.0e-12))

        else:
            raise NotImplementedError(
                f"Schedule '{self.schedule}' no soportado para VP."
            )