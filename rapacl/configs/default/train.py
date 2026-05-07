import os

### Data & Path Settings 

ROOT_DIR = os.path.join(os.path.expanduser("~"), "workspace", "datasets", "rapacl_data")
FEATURE_LIST_PATH = os.path.join(ROOT_DIR, "feature_list.txt")
GENE_LIST_PATH = os.path.join(ROOT_DIR, "var_250genes.json")
TRAIN_SPLIT_CSV = os.path.join(ROOT_DIR, "splits", "train_0.csv")
VAL_SPLIT_CSV = os.path.join(ROOT_DIR, "splits", "test_0.csv")

PROJECT_DIR = os.path.join(os.path.expanduser("~"), "workspace", "RaPaCL")
RADTRANSTAB_PRETRAINED_DIR = os.path.join(PROJECT_DIR, "checkpoints", "radiomics_retrieval", "transtab")
OUTPUT_CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints", "rapacl", "default")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "rapacl", "default")

# LABEL_COL = "target_label"
# ID_COL = "barcode"
NUM_CELLTYPE_CLASSES = 5 

### Model Settings 
# Radiomics TransTab Settings
# commented out is same as default in build_radiomics_learner, so we can directly use the default values without passing them as arguments
NUM_CLASS = NUM_CELLTYPE_CLASSES
HIDDEN_DIM = 128
# NUM_LAYER = 2
PROJECTION_DIM = 384
DROPOUT = 0.1
ACTIVATION = "leakyrelu"
NUM_SUB_COLS = [72, 54, 36, 18, 9, 3, 1]    
APE_DROP_RATE = 0.0

# Pathomics Settings
PATHOMICS_DIM = 1024
PATH_PROJ_HIDDEN_DIM = 512

# Head Settings
RECON_HIDDEN_DIM = 512
CLS_HIDDEN_DIM = 256
GENE_HIDDEN_DIM = 512
HEAD_DROPOUT = 0.1

### Train Settings 
SEED = 0
DEVICE = "cuda:0"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 0
USE_AMP = False
USE_TQDM = False

