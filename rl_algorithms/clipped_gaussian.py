import torch
import math

def elementwise_gaussian_log_pdf(x, mean, var):
    # log N(x|mean,var)
    return -0.5 * torch.log(2 * torch.tensor(torch.pi, device=x.device)) - \
        0.5 * torch.log(var) - ((x - mean) ** 2) / (2 * var)


def _ndtr(a):
    """CDF of the standard normal distribution."""
    x = a / (2 ** 0.5)
    z = x.abs()
    half_erfc_z = 0.5 * torch.erfc(z)
    return torch.where(
        z < (1 / (2 ** 0.5)),
        0.5 + 0.5 * torch.erf(x),
        torch.where(
            x > 0,
            1.0 - half_erfc_z,
            half_erfc_z
        )
    )


def _safe_log(x, epsilon=1e-4):
    """Logarithm function that won't backprop inf to input."""
    x = torch.clamp(x, min=epsilon) # avoid log(0)
    return torch.log(torch.where(x > 0, x, torch.full_like(x, float('nan'), device=x.device)))

def _log_ndtr(x):
    """Log CDF of the standard normal distribution."""
    return torch.where(
        x > 6,
        -_ndtr(-x),
        torch.where(
            x > -14,
            _safe_log(_ndtr(x)),
            -0.5 * x * x - _safe_log(-x) - 0.5 * torch.log(2 * torch.tensor(torch.pi, device=x.device))
        )
    )

def _gaussian_log_cdf(x, mu, sigma):
    """Log CDF of a normal distribution."""
    return _log_ndtr((x - mu) / sigma)


def _gaussian_log_sf(x, mu, sigma):
    """Log SF of a normal distribution."""
    return _log_ndtr(-(x - mu) / sigma)


class ClippedGaussian:
    """Clipped Gaussian distribution."""

    def __init__(self, mean, var, low, high):
        self.mean = mean
        self.var = var
        self.low = low
        self.high = high

    def sample(self):
        unclipped = torch.normal(self.mean, self.var.sqrt())
        return torch.clamp(unclipped, self.low, self.high)

    def log_prob(self, x):
        unclipped_elementwise_log_prob = elementwise_gaussian_log_pdf(x, self.mean, self.var)
        std = self.var.sqrt()
        low_log_prob = _gaussian_log_cdf(self.low, self.mean, std)
        high_log_prob = _gaussian_log_sf(self.high, self.mean, std)

        elementwise_log_prob = torch.where(
            x <= self.low,
            low_log_prob,
            torch.where(
                x >= self.high,
                high_log_prob,
                unclipped_elementwise_log_prob
            )
        )
        return elementwise_log_prob

    def prob(self, x):
        return torch.exp(self.log_prob(x))

    def copy(self):
        return ClippedGaussian(self.mean.clone(), self.var.clone(), self.low.clone(), self.high.clone())