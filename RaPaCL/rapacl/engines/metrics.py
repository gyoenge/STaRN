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

def multiclass_auroc_auprc(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None = None,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Macro one-vs-rest AUROC / AUPRC for multiclass classification.
    Pure PyTorch implementation.

    logits: [N, C]
    target: [N]
    """
    logits = logits.detach().float().cpu()
    target = target.detach().long().cpu()

    probs = torch.softmax(logits, dim=1)

    if num_classes is None:
        num_classes = probs.size(1)

    aurocs = []
    auprcs = []

    for c in range(num_classes):
        y_true = (target == c).float()
        y_score = probs[:, c]

        # Skip class if only positive or only negative exists
        if y_true.sum() == 0 or y_true.sum() == y_true.numel():
            continue

        order = torch.argsort(y_score, descending=True)
        y_true_sorted = y_true[order]

        tp = torch.cumsum(y_true_sorted, dim=0)
        fp = torch.cumsum(1.0 - y_true_sorted, dim=0)

        pos = y_true.sum()
        neg = y_true.numel() - pos

        tpr = tp / (pos + eps)
        fpr = fp / (neg + eps)

        precision = tp / (tp + fp + eps)
        recall = tpr

        # Add start point
        fpr = torch.cat([torch.tensor([0.0]), fpr])
        tpr = torch.cat([torch.tensor([0.0]), tpr])

        recall = torch.cat([torch.tensor([0.0]), recall])
        precision = torch.cat([torch.tensor([1.0]), precision])

        auroc = torch.trapz(tpr, fpr).item()
        auprc = torch.trapz(precision, recall).item()

        aurocs.append(auroc)
        auprcs.append(auprc)

    if len(aurocs) == 0:
        return {
            "auroc": 0.0,
            "auprc": 0.0,
        }

    return {
        "auroc": float(sum(aurocs) / len(aurocs)),
        "auprc": float(sum(auprcs) / len(auprcs)),
    }


#### 


def uniformity(
    z: torch.Tensor,
    t: float = 2.0,
    max_samples: int | None = 5000,
    eps: float = 1e-8,
) -> float:
    """
    Uniformity metric from contrastive learning.

    Lower is generally better, but extremely low values with low effective rank
    may indicate collapse-like behavior.

    z: [N, D]
    """
    z = z.detach().float().cpu()

    if z.size(0) < 2:
        return 0.0

    if max_samples is not None and z.size(0) > max_samples:
        idx = torch.randperm(z.size(0))[:max_samples]
        z = z[idx]

    z = torch.nn.functional.normalize(z, dim=1, eps=eps)

    pairwise_dist_sq = torch.pdist(z, p=2).pow(2)
    score = torch.log(torch.exp(-t * pairwise_dist_sq).mean() + eps)

    return score.item()


def effective_rank(
    z: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Measures how many latent dimensions are effectively used.

    Higher is better.
    Very low value can indicate representation collapse.
    """
    z = z.detach().float().cpu()

    if z.size(0) < 2:
        return 1.0

    z = z - z.mean(dim=0, keepdim=True)

    _, s, _ = torch.linalg.svd(z, full_matrices=False)

    prob = s / (s.sum() + eps)
    entropy = -(prob * torch.log(prob + eps)).sum()
    rank = torch.exp(entropy)

    return rank.item()


def celltype_separability(
    z: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    Simple cell-type separability metric.

    Computes:
    - intra_class_cos: average cosine similarity within same class
    - inter_class_cos: average cosine similarity across different classes
    - separation: intra - inter

    Higher separation is better.
    """
    z = z.detach().float().cpu()
    labels = labels.detach().long().cpu()

    z = torch.nn.functional.normalize(z, dim=1, eps=eps)

    sim = z @ z.T

    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diag = torch.eye(z.size(0), dtype=torch.bool)

    same = same & ~diag
    diff = ~same & ~diag

    if same.sum() == 0:
        intra = 0.0
    else:
        intra = sim[same].mean().item()

    if diff.sum() == 0:
        inter = 0.0
    else:
        inter = sim[diff].mean().item()

    return {
        "intra_class_cos": intra,
        "inter_class_cos": inter,
        "separation": intra - inter,
    }

"""
uniformity: 너무 높으면 embedding이 좁게 뭉침
effective_rank: 낮으면 collapse 의심
separation: 높을수록 cell-type 구조가 잘 분리됨
"""


### 

