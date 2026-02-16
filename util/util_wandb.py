import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback as WBC

# https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html

# wandb_kwargs is the things passed to wandb.init()
# https://docs.wandb.ai/models/ref/python/functions/init

# to login
# https://docs.wandb.ai/models/ref/python/functions/login

entity='soundbendor'
project='mtmidi_sp-full_linearnn'

def login():
    _key = ''
    with open('wandbkey', 'r') as f:
        _tmp = f.readlines()
        _key = _tmp[0].strip()
    wandb.login(key = _key)

def build_kwargs(_config):
    _d = {'entity': entity, 'project': project}
    _d['config'] = _config
    return _d
