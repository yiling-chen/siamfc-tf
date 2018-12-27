import json
from os.path import join
from collections import namedtuple

param_folder = '/mnt/Data-1/Projects/trackers/siamfc-tf/parameters'

def parse_arguments(in_hp={}, in_evaluation={}, in_run={}):

    with open(join(param_folder, 'hyperparams.json')) as json_file:
        hp = json.load(json_file)
    with open(join(param_folder, 'evaluation.json')) as json_file:
        evaluation = json.load(json_file)
    with open(join(param_folder, 'run.json')) as json_file:
        run = json.load(json_file)
    with open(join(param_folder, 'environment.json')) as json_file:
        env = json.load(json_file)
    with open(join(param_folder, 'design.json')) as json_file:
        design = json.load(json_file)                

    for name,value in in_hp.iteritems():
        hp[name] = value
    for name,value in in_evaluation.iteritems():
        evaluation[name] = value
    for name,value in in_run.iteritems():
        run[name] = value
    
    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    run = namedtuple('run', run.keys())(**run)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, run, env, design
