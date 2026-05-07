from __future__ import annotations

import torch 


#### 

def compute_genewise_pcc(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
):
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()

    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    denom = torch.sqrt(
        (pred_c ** 2).sum(dim=0) * (target_c ** 2).sum(dim=0)
    ) + eps

    pcc_per_gene = (pred_c * target_c).sum(dim=0) / denom

    return pcc_per_gene.mean().item(), pcc_per_gene.numpy()


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


#### 

