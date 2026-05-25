import os

### Data & Path Settings 

ROOT_DIR = os.path.join(os.path.expanduser("~"), "workspace", "datasets", "rapacl_data")
FEATURE_LIST_PATH = os.path.join(ROOT_DIR, "feature_list.txt")
GENE_LIST_PATH = os.path.join(ROOT_DIR, "var_250genes.json") # 250 | 50 
SELECT_FOLDS = [0, 1, 2, 3]  # for cross-validation fold selection: default 0~3 

PROJECT_DIR = os.path.join(os.path.expanduser("~"), "workspace", "RaPaCL")
RADTRANSTAB_PRETRAINED_DIR = os.path.join(PROJECT_DIR, "checkpoints", "radiomics_retrieval", "transtab")
OUTPUT_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "rapacl", "default")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "rapacl", "default")

### Model Settings 
# Radiomics TransTab Settings
# commented out is same as default in build_radiomics_learner, so we can directly use the default values without passing them as arguments
NUM_CLASS = NUM_CELLTYPE_CLASSES = 5
HIDDEN_DIM = 128
# NUM_LAYER = 2
PROJECTION_DIM = 384
DROPOUT = 0.1
ACTIVATION = "leakyrelu"
NUM_SUB_COLS = [72, 54, 36, 18, 9, 3, 1] # [72, 54, 36] # [72, 54, 36, 18, 9, 3, 1]    
APE_DROP_RATE = 0.0

# Pathomics Settings
BACKBONE = "densenet121" #"uni" # "densenet121" | "resnet50" | "uni" 
PRETRAINED = True 
PATHOMICS_DIM = 1024
PATH_PROJ_HIDDEN_DIM = 512

# Head Settings
RECON_HIDDEN_DIM = 512
CLS_HIDDEN_DIM = 256
GENE_HIDDEN_DIM = 256 # 256 | 512
HEAD_DROPOUT = 0.1

### Loss Settings
MMCL_LOSS = "multipos_nt_xent" # "symmetric_info_nce" | "multipos_nt_xent"

CONTRASTIVE_TEMPERATURE = 0.25 # 0.07 | 0.15 ~ 0.3 

MMCL_LAMBDA = 1.0 # 0.6 # 0.7 | 1.0 
RECON_LAMBDA = 25.0 #25.0 # 3.0 | 1.0 
CLS_LAMBDA = 1.0 # 1.0

### Train Settings 
SEED = 0
DEVICE = "cuda:0"
BATCH_SIZE = 256 # 128
NUM_WORKERS = 0
USE_AMP = False
USE_TQDM = False

USE_IMAGE_AUGMENTATION = True
USE_PAIR_AUGMENT_BATCH = True

WARMUP_RECON_EPOCHS = 5
MMCL_RAMPUP_EPOCHS = 20 # 15 # 10
STAGE1_EPOCHS = 100 # 30 # 25 # 20 
STAGE2_EPOCHS = 50

LR = 1e-4
GENE_LR = 1e-4
PATH_PROJ_LR = 3e-5 # 1e-5 | 1e-4
PATH_ENCODER_LR = 1e-4 # 1e-5 | 1e-4

WEIGHT_DECAY_STAGE1 = 1e-4
WEIGHT_DECAY_STAGE2 = 1e-3
