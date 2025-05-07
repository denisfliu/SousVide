import torch
import torch.nn.functional as F

def loss_entropy(weights):
    return -torch.mean(torch.sum(weights * torch.log(weights + 1e-8), dim=1))

def loss_load_balance(weights):
    mean_gate = torch.mean(weights, dim=0)  # (E,)
    return torch.var(mean_gate)

class LossFn:
    def __init__(self, alpha=0.1, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, ynn_hat, ynn, ann):
        loss_mse = F.mse_loss(ynn_hat, ynn, reduction='mean')

        if "weights" in ann:
            weights = ann["weights"]
            loss_ent = loss_entropy(weights)
            loss_lbl = loss_load_balance(weights)

            loss = loss_mse + self.alpha*loss_ent + self.beta*loss_lbl
        else:
            loss = loss_mse


        return loss
