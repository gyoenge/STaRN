from rapacl.model.patchenc._densenet import build_densenet121
from rapacl.model.patchenc._resnet import build_resnet50
from rapacl.model.patchenc._uni import build_uni


def build_patch_encoder(backbone: str, pretrained=True,):

    backbone = backbone.lower()

    if backbone == "densenet121":
        return build_densenet121(pretrained)

    elif backbone == "resnet50":
        return build_resnet50(pretrained)

    elif backbone == "uni":
        return build_uni() # pretrained foundation 

    else:
        raise ValueError(f"Unknown backbone: {backbone}")



"""
Usage: 

from model.patchenc import build_patch_encoder
from model.patchenc import constants

encoder, feat_dim = build_patch_encoder(
    backbone=constants.BASE_BACKBONE,
    pretrained=constants.PRETRAINED,
)

print(feat_dim)
"""
