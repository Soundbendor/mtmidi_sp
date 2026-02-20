import util_mail as UM

import numpy as np
import polars as pl
import torch, torch.utils.data as TUD
from sklearn.model_selection import train_test_split

def get_train_test_splits(datadict, train_pct = UM.TRAIN_PCT, test_subpct = UM.TEST_SUBPCT, seed = UM.SEED):
    ret = {}
    if datadict['train_on_middle'] == True:
        num_examples = datadict['num_examples']
        rng = np.random.default_rng(seed)
        _label = datadict['label']
        _idxs = np.arange(num_examples)
        temp_df = pl.DataFrame({'label': datadict['df'][_label], 'idxs': _idxs})
        temp_df = temp_df.sort(_label)
        idxs = temp_df['idxs'].to_numpy()
        test_pct = 1. - train_pct
        first_prop = test_pct * 0.5
        last_prop = 1. - first_prop
        up_to_middle = int(first_prop * num_examples)
        middle_to_end = int(last_prop * num_examples)
        ret['train_idxs'] = idxs[up_to_middle:middle_to_end]
        if test_subpct < 1.:
            ends = np.hstack([idxs[:up_to_middle], idxs[middle_to_end:]])
            validtest = train_test_split(ends, train_size = test_subpct, random_state = seed, shuffle = True)
            ret['test_idxs'] = validtest[0]
            ret['valid_idxs'] = validtest[1]
        else:
            ret['test_idxs'] = np.hstack([idxs[:up_to_middle], idxs[middle_to_end:]])
            ret['valid_idxs'] = np.array([])
    else:
        idxs = np.arange(datadict['num_examples'])
        labels = datadict['df'][datadict['label']].to_numpy()
        cur_train_validtest = train_test_split(idxs, train_size = train_pct, random_state = seed, shuffle = True, stratify = labels)
        ret['train_idxs'] = cur_train_validtest[0]
        if test_subpct < 1.:
            validtest_idxs = cur_train_validtest[1]
            validtest_labels = labels[validtest_idxs]
            cur_validtest = train_test_split(validtest_idxs, train_size = test_subpct, random_state = seed, shuffle = True, stratify = validtest_labels)
            ret['test_idxs'] = cur_validtest[0]
            ret['valid_idxs'] = cur_validtest[1]
        else:
            ret['test_idxs'] = cur_train_validtest[1]
            ret['valid_idxs'] = np.array([])
    return ret





def get_train_test_subsets(dataset_obj, datadict, train_folds = UM.TRAIN_FOLDS, valid_folds =UM.VALID_FOLDS, test_folds = UM.TEST_FOLDS, train_pct = UM.TRAIN_PCT, test_subpct = UM.TEST_SUBPCT, seed = UM.SEED):
    idx_dict = {}
    if datadict['train_on_middle'] == True or len(train_folds) == 0:
        # if train_folds is empty or training on middle, randomize with given pct/subpct splits

        idx_dict = get_train_test_splits(datadict, train_pct = train_pct, test_subpct = test_subpct, seed = seed)
    else:
        # default to given folds
        num_examples = datadict['num_examples']
        _idxs = np.arange(num_examples)
        temp_df = pl.DataFrame({'fold': datadict['df']['fold'], 'idxs': _idxs})
        idx_dict['train_idxs'] = temp_df.filter(pl.col('fold').is_in(train_folds))['idxs'].to_numpy()
        idx_dict['valid_idxs'] = temp_df.filter(pl.col('fold').is_in(valid_folds))['idxs'].to_numpy()
        idx_dict['test_idxs'] = temp_df.filter(pl.col('fold').is_in(test_folds))['idxs'].to_numpy()
    train_subset = TUD.Subset(dataset_obj, idx_dict['train_idxs'])
    valid_subset = None
    test_subset = None
    weights = np.array([])
    train_size = idx_dict['train_idxs'].shape[0]
    valid_size = idx_dict['valid_idxs'].shape[0]
    test_size = idx_dict['test_idxs'].shape[0]
    if datadict['is_balanced'] == False:
        cur_label = datadict['label']
        train_df = datadict['df'][idx_dict['train_idxs']]
        class_amounts = {k:v[0] for (k,v) in train_df[cur_label].value_counts().rows_by_key(cur_label).items()}
        amount_arr = np.array([class_amounts(k) for k in datadict['label_arr']])
        inv_class_prop = train_size/amount_arr
        weights = inv_class_prop/np.max(inv_class_prop)
    if idx_dict['valid_idxs'].shape[0] > 0:
        valid_subset = TUD.Subset(dataset_obj, idx_dict['valid_idxs'])
    if idx_dict['test_idxs'].shape[0] > 0:
        test_subset = TUD.Subset(dataset_obj, idx_dict['test_idxs'])
    ret = {
            'weights': weights,
            'train_subset': train_subset,
            'valid_subset': valid_subset,
            'test_subset': test_subset,
            'train_idxs': idx_dict['train_idxs'],
            'valid_idxs': idx_dict['valid_idxs'],
            'test_idxs': idx_dict['test_idxs'],
            'train_size': train_size,
            'valid_size': valid_size,
            'test_size': test_size,
            'train_folds': train_folds,
            'valid_folds': valid_folds,
            'test_folds': test_folds
            }
    return ret


    
    

