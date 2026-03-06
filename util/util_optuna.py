import os, pickle
import optuna

from . import util_main as UMN
from . import util_constants as UC

linearnn_full_search_space = {'l2_weight_decay_exp': [0, -1, -2, -3, -4], 'dropout': [0]}
linearnn_full_search_space = {'l2_weight_decay_exp': [0, -1, -2, -3, -4], 'dropout': [0, 0.25, 0.5, 0.75]}

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







