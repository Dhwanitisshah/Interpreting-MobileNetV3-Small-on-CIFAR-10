import torch
import torch.nn.functional as F

from .gradcam import GradCAM


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ (Chattopadhay et al., 2018), implemented from scratch.

    Same interface, hooks, target layer, normalization and upsampling as
    `GradCAM` -- only the channel weights differ. Grad-CAM weights every
    spatial location of a channel's gradient equally (a plain spatial mean);
    Grad-CAM++ instead weights each location by alpha, derived from the
    second/third-order gradients of the class score, so channels whose
    evidence is spread over several small regions are not drowned out.

    With the paper's exponential-score trick (S = exp(Y), where Y is the
    class logit) the higher-order derivatives collapse to powers of the
    first-order gradient g = dY/dA, giving the closed form used here:

        alpha_ij = g^2 / (2 g^2 + sum_ab(A_ab) * g^3)
        w_k      = sum_ij alpha_ij * ReLU(g_ij)

    Locations where the denominator is exactly zero (e.g. zero gradient) get
    alpha = 0, as in the reference derivation.
    """

    def _channel_weights(self, activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        grad_2 = gradient.pow(2)
        grad_3 = gradient.pow(3)
        activation_sum = activation.sum(dim=(2, 3), keepdim=True)

        denominator = 2.0 * grad_2 + activation_sum * grad_3
        denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = grad_2 / denominator

        return (alpha * F.relu(gradient)).sum(dim=(2, 3), keepdim=True)
