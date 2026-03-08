import os, pickle
import optuna

from . import util_main as UMN
from . import util_constants as UC

linearnn_full_search_space = {'l2_weight_decay_exp': [0, -1, -2, -3, -4], 'dropout': [0]}
mlp_full_search_space = {'l2_weight_decay_exp': [0, -1, -2, -3, -4], 'dropout': [0, 0.25, 0.5, 0.75]}

# k_div divide activation size by this amount
cae_linear_search_space = {'k_div': [512, 256, 128, 64, 32, 16, 8]}
cae_mlp_search_space = {'k_div': [512, 256, 128, 64, 32, 16, 8]}
def get_layer_search_space(model_size):
    ret = []
    if model_size in set(['small', 'medium', 'large']): 
        num_layers = UC.MODEL_NUM_LAYERS[f'musicgen-{model_size}']
        ret = list(range(num_layers))
    return ret

def create_study_name(parser_args):
    return f'{parser_args.expr_type}-{parser_args.dataset}_{parser_args.model_size}-{parser_args.suffix}'

def record_dict_in_study(studydict, cur_dict):
    flat_dict = UMN.dict_arrayargs_to_str(cur_dict)
    for k,v in flat_dict.items():
        studydict['study'].set_user_attr(k,v)

def create_or_load_study(parser_args, seed=UC.SEED):
    ret = {}

    cur_study_name = create_study_name(parser_args)
    sampler_dir = UMN.by_projpath(UC.SAMPLER_FOLDER, True)
    rdb_dir = UMN.by_projpath(UC.RDB_FOLDER, True)
    sampler_filepath = os.path.join(sampler_dir, f'{cur_study_name}.pkl')
    rdb_filepath = os.path.join(rdb_dir, f'{cur_study_name}.db')
    resuming = False
    cur_sampler = None
    if os.path.exists(rdb_filepath) == True and os.path.exists(sampler_filepath) == True and parser_args.restart_study == False:
        resuming = True
        cur_sampler = pickle.load(open(sampler_file, 'rb'))
    rdb_url = "sqlite:///" + rdb_filepath
    ret['study_name'] = cur_study_name
    ret['sampler_filepath'] = sampler_filepath
    ret['rdb_filepath'] = rdb_filepath
    ret['resuming_study'] = resuming
    ret['study_seed'] = seed

    if cur_sampler == None:
        cur_search_space = {k:v for (k,v) in linearnn_full_search_space.items()}
        cur_search_space['layer_idx'] = get_layer_search_space(parser_args.model_size) 
        cur_sampler = optuna.samplers.GridSampler(cur_search_space, seed=seed)

    ret['study'] = optuna.create_study(study_name=cur_study_name, sampler = cur_sampler, storage=rdb_url, direction=UC.OPT_DIRECTION, load_if_exists = (resuming == True and parser_args.restart_study == False))
    return ret


def get_run_name(configdict, layer_idx, other = None, is_short = False):
    _dataset = configdict['dataset']
    suffix = configdict['suffix']
    _model_size = configdict['model_size']
    if is_short == True:
        _dataset = UC.DATASET_SHORT[_dataset]
        _model_size = UC.MODEL_SIZE_SHORT[_model_size]
    ret = None
    if other == None:
        ret = f'{_dataset}_{_model_size}_l{layer_idx}-{suffix}'
    else:
        ret = f'{_dataset}_{_model_size}_l{layer_idx}_{other}-{suffix}'
    return ret 

#format string to make weight_decay be appendable to a run name
def weight_decay_string_format(weight_decay_exp, is_short = False ):
    wd_int = abs(int(weight_decay_exp))
    if is_short == False:
        return f'weightdecay{wd_int}'
    else: 
        return f'wd{wd_int}'


#format string to make dropout be appendable to a run name
def dropout_string_format(dropout, is_short = False):
    dropout_int = int(dropout * 100)
    if is_short == False:
        return f'dropout{dropout_int}'
    else:
        return f'do{dropout_int}'


def get_run_and_short_names(configdict, layer_idx, name_params):
    other_long_arr = []
    other_short_arr = []
    if 'l2_weight_decay_exp' in name_params.keys():
        wd_long = weight_decay_string_format(name_params['l2_weight_decay_exp'] , is_short = False)
        wd_short = weight_decay_string_format(name_params['l2_weight_decay_exp'] , is_short = True)
        other_long_arr.append(wd_long)
        other_short_arr.append(wd_short)
    if 'dropout' in name_params.keys():
        do_long = dropout_string_format(name_params['dropout'],is_short = False)
        do_short = dropout_string_format(name_params['dropout'],is_short = True)
        other_long_arr.append(do_long)
        other_short_arr.append(do_short)
    other_long = '_'.join(other_long_arr)
    other_short = '_'.join(other_short_arr)
    run_name = get_run_name(configdict, layer_idx, other = other_long, is_short = False) 
    short_name = get_run_name(configdict, layer_idx, other = other_short, is_short = True)
    return run_name, short_name



