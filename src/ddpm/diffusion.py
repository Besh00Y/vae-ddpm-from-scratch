import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    """Pick a[t] for each sample in the batch and reshape to broadcast over images."""
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer(
            "posterior_variance", betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        )

    # ---- forward process: q(x_t | x_0) ----
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alpha_bar, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alpha_bar, t, x0.shape) * noise
        )

    # ---- training loss: predict the noise ----
    def loss(self, x0):
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        return F.mse_loss(pred, noise)

    # ---- reverse process: p(x_{t-1} | x_t) ----
    @torch.no_grad()
    def p_sample(self, model, x, t):
        t_batch = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        eps = model(x, t_batch)

        coef = extract(self.betas, t_batch, x.shape) / extract(
            self.sqrt_one_minus_alpha_bar, t_batch, x.shape
        )
        mean = extract(self.sqrt_recip_alphas, t_batch, x.shape) * (x - coef * eps)

        if t == 0:
            return mean
        var = extract(self.posterior_variance, t_batch, x.shape)
        return mean + var.sqrt() * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, n, shape, device, model=None):
        """shape = (channels, H, W). Pass an EMA model via `model` if you have one."""
        model = model if model is not None else self.model
        model.eval()
        x = torch.randn(n, *shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)