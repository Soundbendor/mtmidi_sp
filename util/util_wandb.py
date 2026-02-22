import os

import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback as WBC

import util_main as UM

# https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html

# wandb_kwargs is the things passed to wandb.init()
# https://docs.wandb.ai/models/ref/python/functions/init

# to login
# https://docs.wandb.ai/models/ref/python/functions/login

entity='soundbendor'

def login():
    _key = ''
    with open('wandbkey', 'r') as f:
        _tmp = f.readlines()
        _key = _tmp[0].strip()
    wandb.login(key = _key)

def build_config(parser_args, datadict, subsetdict):
    _config = {k:v for (k,v) in vars(parser_args).items()}
    _config['num_epochs'] = UM.NUM_EPOCHS
    if parser_args.expr_type != 'standard_scaler':
        _config['early_stopping_check_interval'] = UM.EARLY_STOPPING_CHECK_INTERVAL
        _config['early_stopping_boredom'] = UM.EARLY_STOPPING_BOREDOM
    _config['train_folds'] = subsetdict['train_folds']
    _config['valid_folds'] = subsetdict['valid_folds']
    _config['test_folds'] = subsetdict['test_folds']
    _config['is_balanced'] = datadict['is_balanced']
    _config['use_weights'] = subsetdict['weights'].shape[0] > 0
    return _config


def build_kwargs(_config, expr_type):
    _d = {'entity': entity, 'project': f'mtmidi_sp-{expr_type}'}
    _d['config'] = _config
    return _d

