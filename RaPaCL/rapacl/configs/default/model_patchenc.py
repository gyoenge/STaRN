import os

PRETRAINED = True
DEVICE = "cuda:0"
PROJECT_DIR = os.path.join(os.path.expanduser("~"), "workspace", "RaPaCL")

### UNI 
UNI_VERSION = "vit_large_patch16_224"
UNI_IMG_SIZE = 224
UNI_PATCH_SIZE = 16 
UNI_CKPT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "UNI", f"{UNI_VERSION}.bin") 
