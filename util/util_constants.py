from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTACTS_FOLDER = 'postacts'
SAMPLER_FOLDER = 'samplers'
SCALERS_FOLDER = 'scalers'
SCALERS_DOC_FOLDER = 'scalers_doc'
CM_FOLDER = 'cm'
RESULTS_FOLDER = 'res'
MODELS_FOLDER = 'model_models'
RDB_FOLDER = 'rdb'
NUM_FOLDS = 20
EARLY_STOPPING_CHECK_INTERVAL = 1
EARLY_STOPPING_BOREDOM = 10
MEMMAP = True
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 10.**(-3)
# no l2 weight decay (set to -2 in original which meant turn off)
DATALOADER_SHUFFLE = True
TRAIN_FOLDS = list(range(1,15))
VALID_FOLDS = list(range(15,18))
TEST_FOLDS = list(range(18,21))
IS_64BIT = False
SEED = 39
TRAIN_PCT = 0.7
TEST_SUBPCT = 0.5
OPT_DIRECTION = 'maximize'
STANDARD_SCALER_CONSTANT_FEATURE_MASK = True
LINEARNNPROBE_INITIAL_DROPOUT = False

MLPPROBE_INITIAL_DROPOUT = False
MLPPROBE_HIDDEN_DROPOUT = True
MLPPROBE_HIDDEN_DIM_MULT = 0.5

CAE_INIT_TEMP = 10.
CAE_FINAL_TEMP = 0.01

# for less than 11 classes
CM_FIGSIZE_S = (5,5)
# for 11 classes and up
CM_FIGSIZE_M = (7,7)
# for a lot of classses
CM_FIGSIZE_L = (18,15)

SHARE_PATH = os.path.join(os.sep, 'nfs','hpc', 'share', 'kwand') 
WANDB_PATH = os.path.join(os.sep, 'nfs','guille', 'eecs_research', 'soundbendor', 'kwand', 'wandb') 


EXPR_PRETTY_NAMES = {'linearnn_full': 'Linear NN (full)', 'mlp_full': 'MLP NN (full)'}
MUSICGEN_SIZES = ["small", "medium", "large"]

EXPR_SHORT = {"linearnn_full": "lnf", "standard_scaler": "sts", 'mlp_full': 'mlpf'}
SIZES_SHORT = {"small": "s", "medium": "m", "large": "l"}

DATASET_SHORT = {"polyrhythms": "pl",
                 "dynamics": "dyn",
                 "seventh_chords": "ch7",
                 "mode_mixture": "mm",
                 "secondary_dominants": "sd",
                 "tempos": "tpo",
                 "time_signatures": "ts",
                 "chords": "chd",
                 "notes": "not",
                 "scales": "scl",
                 "intervals": "ivl",
                 "simple_progressions": "spg"
                 }

DATASET_PRETTY = {"polyrhythms": "Polyrhythms",
                 "dynamics": "Dynamics",
                 "seventh_chords": "Seventh Chords",
                 "mode_mixture": "Mode Mixture",
                 "secondary_dominants": "Secondary Dominants",
                 "tempos": "Tempos",
                 "time_signatures": "Time Signatures",
                 "chords": "Chords",
                 "notes": "Notes",
                 "scales": "Scales",
                 "intervals": "Intervals",
                 "simple_progressions": "Simple Progressions"
                 }



MODEL_SIZE_SHORT = {"small": "sm", "medium": "med", "large": "lg"}
# https://github.com/huggingface/transformers/blob/80996194bec45b16d4472a099e64b57e049bc6fd/src/transformers/models/musicgen/convert_musicgen_transformers.py#L120
FFN_DIM = {"musicgen-small": 1024 * 4, "musicgen-medium": 1536 * 4, "musicgen-large": 2048 * 4}

# this time not using initial embeddings
MODEL_NUM_LAYERS = {"musicgen-small": 24, "musicgen-medium": 48, "musicgen-large": 48}

### porting a lot of old code from mtmidi

MUSICGEN_SR = 32000
# same as mtmidi
# but secondary_dominant -> secondary_dominants
# modemix_chordprog -> mode_mixture
# chords7 -> seventh_chords
SYNTHEORY_PLUS_DATASETS = set(['polyrhythms', 'dynamics', 'seventh_chords', 'secondary_dominants', 'mode_mixture'])

SYNTHEORY_DATASETS = set(['tempos', 'time_signatures', 'chords', 'notes', 'scales', 'intervals', 'simple_progressions'])
CHORDPROG_DATASETS = set(['secondary_dominant', 'modemix_chordprog', 'simple_progressions'])

MODELS = ['musicgen-small', 'musicgen-medium', 'musicgen-large']

#datasets that are regression
REG_DATASETS = set(['tempos'])
# datasets to train on middle on
TOM_DATASETS = set(['tempos'])
ALL_DATASETS = SYNTHEORY_DATASETS.union(SYNTHEORY_PLUS_DATASETS)

