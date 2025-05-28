import torch
import torch.nn.functional as F

def loss_entropy(weights):
    return -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))

def loss_load_balance(weights):
    mean_gate = torch.mean(weights, dim=0)  # (E,)
    return torch.var(mean_gate)

class LossFn:
    def __init__(self, alpha=0.1, beta=0.1):
        pass

    def __call__(self, ypd:dict[str,torch.Tensor], ylb:dict[str,torch.Tensor]):

        loss = 0.0
        for key in ylb.keys():
            loss += F.mse_loss(ypd[key], ylb[key], reduction='mean')

        return loss
