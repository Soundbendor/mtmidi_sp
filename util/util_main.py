import os
import numpy as np
import datasets as HFDS
import librosa

POSTACTS_FOLDER = 'postacts'
NUM_FOLDS = 20
EARLY_STOPPING_CHECK_INTERVAL = 1
EARLY_STOPPING_BOREDOM = 10
MEMMAP = True
NUM_EPOCHS = 100
TRAIN_FOLDS = list(range(1,15))
VALID_FOLDS = list(range(15,18))
TEST_FOLDS = list(range(18,21))
SEED = 39
TRAIN_PCT = 0.7
TEST_SUBPCT = 0.5
# https://github.com/huggingface/transformers/blob/80996194bec45b16d4472a099e64b57e049bc6fd/src/transformers/models/musicgen/convert_musicgen_transformers.py#L120
ffn_dim = {"musicgen-small": 1024 * 4, "musicgen-medium": 1536 * 4, "musicgen-large": 2048 * 4}

# this time not using initial embeddings
model_num_layers = {"musicgen-small": 24, "musicgen-medium": 48, "musicgen-large": 48}

### porting a lot of old code from mtmidi
share_path = os.path.join(os.sep, 'nfs','hpc', 'share', 'kwand') 

model_sr = 32000
# same as mtmidi
# but secondary_dominant -> secondary_dominants
# modemix_chordprog -> mode_mixture
# chords7 -> seventh_chords
new_datasets = set(['polyrhythms', 'dynamics', 'seventh_chords', 'secondary_dominants', 'mode_mixture'])

hf_datasets = set(['tempos', 'time_signatures', 'chords', 'notes', 'scales', 'intervals', 'simple_progressions'])
chordprog_datasets = set(['secondary_dominant', 'modemix_chordprog'])

models = ['musicgen-small', 'musicgen-medium', 'musicgen-large']

#datasets that are regression
reg_datasets = set(['tempos'])
# datasets to train on middle on
tom_datasets = set(['tempos'])
all_datasets = hf_datasets.union(new_datasets)

# https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
# takes mean of stereo channels (doesn't rely on loading as mono)
# normalizes via numpy (divide by max)
def load_wav(fpath, dur = 4., normalize = False, sr=32000):
    snd, load_sr = librosa.load(fpath, duration = dur, mono = True, sr=sr)
    if normalize == False:
        return snd
    else:
        return librosa.util.normalize(snd)

def by_projpath(subpath=None,make_dir = False, other_projdir = ''):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    if len(other_projdir) > 0:
        cur_path = other_projdir
    if subpath != None:
        cur_path = os.path.join(cur_path, subpath)
        if os.path.exists(cur_path) == False and make_dir == True:
            os.makedirs(cur_path)
    return cur_path

def load_syntheory_train_dataset(ds_name, streaming = True):
    cur_ds =  HFDS.load_dataset("meganwei/syntheory", ds_name, split = 'train', streaming = streaming)
    return cur_ds

### new stuff
def get_hf_model_str(model_size):
    model_str = f"facebook/musicgen-{model_size}" 
    return model_str


def get_model_postacts_path(model_size, dataset='polyrhythms', return_relative = False, make_dir = False, other_projdir = '', fold_num = -1):
    datapath = None
    if return_relative == False:
        postactpath = by_projpath(POSTACTS_FOLDER,make_dir = make_dir, other_projdir = other_projdir)
        datapath = os.path.join(postactpath, dataset)
    else:
        datapath = POSTACTS_FOLDER
    modelpath = os.path.join(datapath, f'musicgen-{model_size}')
    if fold_num > 0:
        modelpath = os.path.join(modelpath, f'fold_{fold_num}')
    if os.path.exists(modelpath) == False and make_dir == True:
            os.makedirs(modelpath)
    return modelpath

# returns list of directories/files in path
# added complexity with fold folders
# fold_num 0 means search over all 20 folds, -1 means don't care about folds

# original just listed filenames,
# now we need actual paths because folds
def filepath_list(file_dir, fold_num=-1, ignore_exts = set(['.csv'])):
    files = []
    if fold_num < 0:
        files = [os.path.join(file_dir, x) for x in os.listdir(file_dir) if os.path.splitext(x)[-1] not in ignore_exts]
    elif fold_num == 0:
        for i in range(1,NUM_FOLDS+1):
            fold_dir = os.path.join(file_dir, f'fold_{i}')
            cur_files = [os.path.join(fold_dir, x) for x in os.listdir(fold_dir) if os.path.splitext(x)[-1] not in ignore_exts]
            files += cur_files
    else:
        fold_dir = os.path.join(file_dir, f'fold_{fold_num}')
        files = [os.path.join(fold_dir, x) for x in os.listdir(fold_dir) if os.path.splitext(x)[-1] not in ignore_exts]
    return files


# added complexity with fold folders
# fold_num 0 means search over all 20 folds, -1 means don't care about folds
def get_sorted_contents(cur_dir, is_relative = True,fold_num = -1):
    file_dir = None
    if is_relative == True:
        file_dir = by_projpath(subpath=cur_dir, make_dir = False)
    else:
        file_dir = cur_dir
    files = filepath_list(file_dir, fold_num=fold_num)
    file_sort = sorted(files, key = os.path.getmtime)
    return file_sort

# removes latest file because probably incomplete
# added complexity with fold folders
# fold_num 0 means all 20 folds, -1 means we don't care about folds
def remove_latest_file(cur_dir, is_relative = True, fold_num = -1):
    file_sort = get_sorted_contents(cur_dir, is_relative=is_relative, fold_num = fold_num)
    if len(file_sort) > 0:
        os.remove(file_sort[-1])
    return file_sort[:-1]

# gets filename
def get_basename(file, with_ext = True):
    if with_ext == True:
        return os.path.basename(file)
    else:
        return os.path.splitext(os.path.basename(file))[0]

# given a filepath, get the fold number (full path)
# an example input: /nfs/hpc/share/kwand/syntheory_plus/dynamics/fold_9/subp-3_pp-mf_Woodblock_beat4-6_rvb1_off431.wav
def get_fold_num_from_filepath(filepath):
    pathsplit = os.path.split(filepath)[0]
    fold_folder = pathsplit.split(os.sep)[-1]
    fold_num = int(fold_folder.split("_")[-1])
    return fold_num

    
# replace extension from path
def ext_replace(old_path, new_ext = 'pt'):
    fsplit = '.'.join(old_path.split('.')[:-1])
    outname = fsplit
    if len(new_ext) > 0:
        outname = f'{fsplit}.{new_ext}'
    else:
        outname = f'{fsplit}'
    return outname

def get_postacts_shape(model_size):
    model_str = f'musicgen-{model_size}'
    return (model_num_layers[model_str], ffn_dim[model_str])


# use_shape argument overrides shape getting (useful for baselines)
def get_postacts_file(model_size, dataset='polyrhythms', fname='', write = True, use_64bit = True, use_shape = None, other_projdir = '', fold_num = -1):
    modelpath = get_model_postacts_path(model_size, dataset = dataset, return_relative = False, make_dir = write, other_projdir = other_projdir, fold_num = fold_num)
    fpath = os.path.join(modelpath, fname)
    fp = None
    dtype = 'float32'
    mode = 'r'
    shape = None
    if use_shape == None:
        shape = get_postacts_shape(model_size)
    else:
        shape = use_shape
    if use_64bit == True:
        dtype = 'float64'
    if write == True:
        if os.path.isfile(fpath) == True:
            mode = 'r+'
        else:
            mode = 'w+'
    fp = np.memmap(fpath, dtype = dtype, mode=mode, order='C', shape=shape)
    return fp

def save_npy(save_arr, fname, model_size, dataset='polyrhythms', make_dir = True, other_projdir = '', fold_num = -1):
    modelpath = get_model_postacts_path(model_shorthand, dataset = dataset, return_relative = False, make_dir = make_dir, other_projdir = other_projdir, fold_num = fold_num)
    fpath = os.path.join(modelpath, fname)
    np.save(fpath, save_arr, allow_pickle = True)

def dict_arrayargs_to_str(cur_dict):
    ret = {}
    for (k,v) in cur_dict.items():
        if isinstance(v, np.ndarray) or isinstance(v, list) or isinstance(v, tuple):
            cur_str = ','.join([str(x) for x in v])
            ret[k] = cur_str
        else:
            ret[k] = v
    return ret
