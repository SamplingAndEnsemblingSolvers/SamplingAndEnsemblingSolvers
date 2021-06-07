import random
import os

import argparse
from argparse import Namespace
import torch
import torchvision.transforms as transforms
import numpy as np
import wandb
import copy
import pickle
from decimal import Decimal
import itertools
from collections import defaultdict

import sys
sys.path.append('../../../')
import sopa.src.models.odenet_cifar10.layers as cifar10_models
from sopa.src.models.odenet_cifar10.utils import *
from sopa.src.solvers.utils import create_solver, noise_params, create_solver_ensemble_by_noising_params
from sopa.src.models.odenet_cifar10.data import get_cifar10_test_loader


import robustbench as rb 
import eagerpy as ep
import foolbox as fb
import MegaAdversarial.src.attacks as ma


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()

parser.add_argument('--wandb_project_name', type=str, default=None)
parser.add_argument('--torch_dtype', type=str, default='float32')


###############################
### ROBUSTBENCH CHECKPOINS ###
###############################

parser.add_argument('--robustbench_checkpoints_root', type=str,
                    default="./checkpoints_robustbench")

###############################
### PARAMS FOR SOURCE MODEL ###
###############################


parser.add_argument('--source_name', type=str, default=None)
parser.add_argument('--source_checkpoint_wnb_run_path', type=str, default=None)
parser.add_argument('--source_checkpoint_wnb_name', type=str, default=None)
parser.add_argument('--source_checkpoint_path', type=str, default=None)


parser.add_argument('--source_solvers', type=str, default=None,
                    help='Each solver is represented with (method,parameterization,n_steps,step_size,u0,v0) \n' +
                         'If the solver has only one parameter u0, set v0 to -1; \n' +
                         'n_steps and step_size are exclusive parameters, only one of them can be != -1, \n'
                         'If n_steps = step_size = -1, automatic time grid_constructor is used \n;'
                         'For example, --source_solvers rk4,uv,2,-1,0.3,0.6;rk3,uv,-1,0.1,0.4,0.6;rk2,u,4,-1,0.3,-1')

# Params for MetaODEBlock in switch mode
parser.add_argument('--source_solver_mode', type=str, default=None, choices=['switch', 'ensemble', 'standalone'],
                    help='MetaODE block uses this regime to perform forward pass')

# Params for MetaODEBlock in ensemble(averaging ODE outputs) mode
parser.add_argument('--source_switch_probs', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help='--source_switch_probs 0.8,0.1,0.1 \n' +
                         'This argument is used when source_solver_mode="switch"',)

parser.add_argument('--source_ensemble_prob', type=float, default=1.,
                    help='This argument is used when source_solver_mode="ensemble"')

parser.add_argument('--source_ensemble_weights', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help='--source_ensemble_weights 0.6,0.2,0.2 \n' +
                         'This argument is used when source_solver_mode="ensemble"')

# Params for MetaNode in ensemble(averaging probabilities) mode
parser.add_argument('--source_model_ensemble_size', type=int, default=1, help='''Number of models in the ensemble''')
parser.add_argument('--source_model_ensemble_average_mode', type=str, choices=['probs', 'perturbs', None])


###############################
### PARAMS FOR TARGET MODEL ###
###############################

parser.add_argument('--target_name', type=str, default=None)
parser.add_argument('--target_checkpoint_wnb_run_path', type=str, default=None)
parser.add_argument('--target_checkpoint_wnb_name', type=str, default=None)
parser.add_argument('--target_checkpoint_path', type=str, default=None)


parser.add_argument('--target_solvers', type=str, default=None,
                    help='Each solver is represented with (method,parameterization,n_steps,step_size,u0,v0) \n' +
                         'If the solver has only one parameter u0, set v0 to -1; \n' +
                         'n_steps and step_size are exclusive parameters, only one of them can be != -1, \n'
                         'If n_steps = step_size = -1, automatic time grid_constructor is used \n;'
                         'For example, --target_solvers rk4,uv,2,-1,0.3,0.6;rk3,uv,-1,0.1,0.4,0.6;rk2,u,4,-1,0.3,-1')

# Params for MetaODEBlock in switch mode
parser.add_argument('--target_solver_mode', type=str, default=None, choices=['switch', 'ensemble', 'standalone'],
                    help='MetaODE block uses this regime to perform forward pass')

# Params for MetaODEBlock in ensemble(averaging ODE outputs) mode
parser.add_argument('--target_switch_probs', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help='--target_switch_probs 0.8,0.1,0.1 \n' +
                         'This argument is used when source_solver_mode="switch"',)

parser.add_argument('--target_ensemble_prob', type=float, default=1.,
                    help='This argument is used when source_solver_mode="ensemble"')

parser.add_argument('--target_ensemble_weights', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help='--target_ensemble_weights 0.6,0.2,0.2 \n' +
                         'This argument is used when source_solver_mode="ensemble"')

# Params for MetaNode in ensemble(averaging probabilities) mode
parser.add_argument('--target_model_ensemble_size', type=int, default=1, help='''Number of models in the ensemble''')


###############################
####### LOADERS PARAMS ########
###############################

parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--pin_memory', type=int, default=True)
parser.add_argument('--download', type=int, default=0)
parser.add_argument('--val_perc', type=float, default=0.1,
                    help='Percentage split of the training set used for the validation set. \n'
                         'Should be a float in the range [0, 1].')


###############################
## ADVERSARIAL ATTACKS PARAMS #
###############################

parser.add_argument('--attack_mode',
                    default="fgsm",
                    choices=["clean", "fgsm", "pgd", "deepfool"],
                    help='''Adversarsarial testing mode''')
parser.add_argument('--epsilons', type=str, default=None,
                    help='Epsilons for adversarial testing multiplied by 255. Example: --epsilons "2,3,8,16,32"')
parser.add_argument('--pgd_lr', type=float, default=2/255.,
                    help='lr of PGD for testing')
parser.add_argument('--pgd_niters', type=int, default=7,
                    help='Number of PGD iterations for testing')

args, unknown_args = parser.parse_known_args()


ROBUSTBENCH_NAMES = ['Wong2020Fast', 'Sehwag2021Proxy_R18', 'Carmon2019Unlabeled']

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)



def load_model(model_name,
               checkpoint_wnb_run_path=None,
               checkpoint_wnb_name=None,
               robustbench_checkpoints_dir=None,
               checkpoint_path=None):
    
    if model_name in ROBUSTBENCH_NAMES:
        model = load_model_robustbench(model_name, robustbench_checkpoints_dir)
        
    elif (checkpoint_wnb_run_path is not None) and (checkpoint_wnb_name is not None) or (checkpoint_path is not None):
        model = load_model_metanode(checkpoint_wnb_run_path, checkpoint_wnb_name, checkpoint_path)
        
    else:
        raise NotImplementedError(f'{model_name} checkpoint can not be loaded')
        
    model.eval()
    return model
        

def load_model_robustbench(model_name, robustbench_checkpoints_dir=None):
    model = rb.utils.load_model(model_name=model_name,
                                 threat_model='Linf',
                                 model_dir=robustbench_checkpoints_dir)
    return model


def load_model_metanode(checkpoint_wnb_run_path, checkpoint_wnb_name, checkpoint_path=None):
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
    elif (checkpoint_wnb_run_path is not None) and (checkpoint_wnb_name is not None):
        checkpoint = torch.load(wandb.restore(checkpoint_wnb_name, run_path=f"/{checkpoint_wnb_run_path}/").name,
                                map_location='cpu')
        
    model = init_model_metanode(checkpoint)
    
    
    checkpoint=None
    torch.cuda.empty_cache()
    return model


def init_model_metanode(checkpoint):
    config = Namespace(**checkpoint['wandb_config'])
    
    norm_layers = (get_normalization(config.normalization_resblock),
                   get_normalization(config.normalization_odeblock),
                   get_normalization(config.normalization_bn1))
    param_norm_layers = (get_param_normalization(config.param_normalization_resblock),
                         get_param_normalization(config.param_normalization_odeblock),
                         get_param_normalization(config.param_normalization_bn1))
    act_layers = (get_activation(config.activation_resblock),
                  get_activation(config.activation_odeblock),
                  get_activation(config.activation_bn1))

    model = getattr(cifar10_models, config.network)(norm_layers, param_norm_layers, act_layers,
                                                    config.in_planes, is_odenet=config.is_odenet)
    model.load_state_dict(checkpoint['model'])
    return model





def get_input_shift_scale(model_name):
    input_shift = (0., 0., 0.) if model_name in ROBUSTBENCH_NAMES else cifar10_mean
    input_scale = (1., 1., 1.) if model_name in ROBUSTBENCH_NAMES else cifar10_std
    
    return input_shift, input_scale
    
    
    
def get_solvers_solver_options_arr(solvers_config, model_ensemble_config, dtype, device):
    
    if solvers_config.solvers is not None:
        splitted_solvers = [tuple(map(lambda iparam: str(iparam[1]) if iparam[0] <= 1 else (
            int(iparam[1]) if iparam[0] == 2 else (
                float(iparam[1]) if iparam[0] == 3 else Decimal(iparam[1]))),
                                      enumerate(item.split(',')))) for item in solvers_config.solvers.strip().split(';')
                           ]
    else:
        splitted_solvers = []
        
    solvers = [create_solver(*solver_params, dtype=dtype, device=device) for solver_params in splitted_solvers]
    for solver in solvers:
        solver.freeze_params()
    
    if not solvers:
        solvers_solver_options_arr = {}
    else:
        solver_options = Namespace(**{key: vars(solvers_config)[key] for key in ['solver_mode',
                                                                           'switch_probs',
                                                                           'ensemble_prob',
                                                                           'ensemble_weights']})
        if model_ensemble_config.model_ensemble_size == 1:
            # one forward with many solvers
            solvers_solver_options_arr = [{'solvers': solvers,
                                           'solver_options': solver_options}]
        else:
            # many forwards, one solver for each forward
            solvers_solver_options_arr = [{'solvers': [solver],
                                           'solver_options': solver_options} for solver in solvers]
    return solvers_solver_options_arr



def generate_advs_foolbox(images, labels, attack, epsilons, fmodels, solvers_solver_options_arr=None,):
    
    raw_advs_ensemble = [0 for _ in range(len(epsilons))]
    
    if solvers_solver_options_arr:
        for n, (fmodel, solvers_solver_options) in enumerate(
            itertools.zip_longest(fmodels, solvers_solver_options_arr, fillvalue=fmodels[0])):
            
            raw_advs, _, _ = attack(fmodel, images, labels, epsilons=epsilons, **solvers_solver_options)
            
            for i in range(len(epsilons)):
                raw_advs_ensemble[i] += raw_advs[i]
                
    else:
        for n, fmodel in enumerate(fmodels):
            raw_advs, _, _ = attack(fmodel, images, labels, epsilons=epsilons)

            for i in range(len(epsilons)):
                raw_advs_ensemble[i] += raw_advs[i]
            
    clipped_advs_ensemble = []
    for i, epsilon in enumerate(epsilons):
        raw_advs_ensemble[i] /= (n+1)
        clipped_advs_ensemble.append(attack.distance.clip_perturbation(images, raw_advs_ensemble[i], epsilon))
        
    return clipped_advs_ensemble


def generate_advs_megaadversarial(images, labels, attack, epsilons, models, solvers_solver_options_arr=[]):
    attack.models = models

    clipped_advs = []
    for epsilon in epsilons:
        
        attack.eps = epsilon

        clipped_adv, _ = attack(images, labels, solvers_solver_options_arr)
        clipped_advs.append(clipped_adv.cpu())
    
    return clipped_advs


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

def predict_probs(x, models, solvers_solver_options_arr=None, ):
    
    probs_ensemble = 0

    with torch.no_grad():
        if solvers_solver_options_arr:
            for n, (model, solvers_solver_options) in enumerate(
                itertools.zip_longest(models, solvers_solver_options_arr, fillvalue=models[0])):

                logits = model(x, **solvers_solver_options)
                probs = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
                probs_ensemble = probs_ensemble + probs

        else:
            for n, model in enumerate(models):
                logits = model(x)
                probs = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
                probs_ensemble = probs_ensemble + probs

        probs_ensemble /= (n + 1)
        return probs_ensemble
    
    
def attack_validate(attack,
                    epsilons,
                    test_loader,
                    source_model_ensemble_config,
                    source_solvers_solver_options_arr,
                    target_solvers_solver_options_arr,
                    source_model=None,
                    source_fmodel=None,
                    target_model=None,
                    data_inv_normalize=None,
                    target_normalize=None,):
    
    for p in target_model.parameters():
        device = p.device
        break

    total_correct = torch.zeros(len(epsilons))

    for idx, (image, label) in enumerate(test_loader):
        target_class = np.argmax(one_hot(np.array(label.cpu().numpy()), 10), axis=1)

        image = data_inv_normalize(image.to(device)) # image in [0, 1]
        label = label.to(device)

        if attack is not None:
            if source_model_ensemble_config.model_ensemble_size > 1 and source_model_ensemble_config.average_mode == 'probs':
                clipped_advs = generate_advs_megaadversarial(image, label, attack, epsilons,
                                                             [source_model], source_solvers_solver_options_arr)
            else:
                images, labels = ep.astensors(image, label)
                clipped_advs = generate_advs_foolbox(images, labels, attack, epsilons,
                                                     [source_fmodel], source_solvers_solver_options_arr)
        else:
            clipped_advs = [image]
            epsilons = [0.]

        for i, epsilon in enumerate(epsilons):
            with torch.no_grad():
                x = torch.tensor(clipped_advs[i].numpy()).to(device)
                x = target_normalize(x)

                probs_ensemble = predict_probs(x, [target_model], target_solvers_solver_options_arr)
                predicted_class = np.argmax(probs_ensemble, axis=1)

                total_correct[i] += np.sum(predicted_class == target_class)

    res = total_correct / (len(test_loader) * test_loader.batch_size)
    return res


if __name__=="__main__":
    wandb.init(project="sampling-ensembling-solvers-node", anonymous="allow",)
    wandb.config.update(args)
    
    if wandb.config.torch_dtype == 'float64':
        dtype = torch.float64
    elif wandb.config.torch_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('torch_type should be either float64 or float32')
        
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    epsilons = [int(eps)/255. for eps in wandb.config.epsilons.strip().split(',')]

    test_loader = get_cifar10_test_loader(batch_size=wandb.config.batch_size,
                                      data_root=wandb.config.data_root,
                                      num_workers=wandb.config.num_workers,
                                      pin_memory=wandb.config.pin_memory,
                                      shuffle=False,
                                      download=wandb.config.download)
    
    
    source_solvers_config = Namespace(**{'solvers': wandb.config.source_solvers,
                                         'solver_mode': wandb.config.source_solver_mode,
                                         'switch_probs': wandb.config.source_switch_probs,
                                         'ensemble_prob': wandb.config.source_ensemble_prob,
                                         'ensemble_weights': wandb.config.source_ensemble_weights,
                                        })
    source_model_ensemble_config = Namespace(**{'model_ensemble_size': wandb.config.source_model_ensemble_size,
                                                'average_mode': wandb.config.source_model_ensemble_average_mode,})



    target_solvers_config = Namespace(**{'solvers': wandb.config.target_solvers,
                                         'solver_mode': wandb.config.target_solver_mode,
                                         'switch_probs': wandb.config.target_switch_probs,
                                         'ensemble_prob': wandb.config.target_ensemble_prob,
                                         'ensemble_weights': wandb.config.target_ensemble_weights,
                                        })
    target_model_ensemble_config = Namespace(**{'model_ensemble_size': wandb.config.target_model_ensemble_size,})



    # Load source and target models.
    source_model = load_model(model_name=wandb.config.source_name,
                              robustbench_checkpoints_dir=wandb.config.robustbench_checkpoints_root,
                              checkpoint_wnb_run_path=wandb.config.source_checkpoint_wnb_run_path,
                              checkpoint_wnb_name=wandb.config.source_checkpoint_wnb_name,
                              checkpoint_path=wandb.config.source_checkpoint_path)
    target_model = load_model(model_name=wandb.config.target_name,
                              robustbench_checkpoints_dir=wandb.config.robustbench_checkpoints_root,
                              checkpoint_wnb_run_path=wandb.config.target_checkpoint_wnb_run_path,
                              checkpoint_wnb_name=wandb.config.target_checkpoint_wnb_name,
                              checkpoint_path=wandb.config.target_checkpoint_path)


    # Define inverse transformation for images from data loader: converts images to [0, 1] range.
    dataset_mean = cifar10_mean if cifar10_mean is not None else (0., 0., 0.)
    dataset_std = cifar10_std if cifar10_std is not None else (1., 1., 1.)
    data_inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(dataset_mean, dataset_std)], std=[1/s for s in dataset_std])

    # Get shifting and scaling factors that transform images from [0, 1] range
    # to the range of source/target model inputs (i.e. inputs to the forward pass). 
    source_input_shift, source_input_scale = get_input_shift_scale(wandb.config.source_name)
    target_input_shift, target_input_scale = get_input_shift_scale(wandb.config.target_name)

    # Define preprocessing for FoolBox attack generator:
    # converts images from [0, 1] range to the range of source model's inputs.
    source_preprocessing = dict(mean=source_input_shift, std=source_input_scale, axis=-3)

    # Define image transformation before validation:
    # converts images from [0, 1] range to the range of target model's inputs.
    target_normalize = transforms.Normalize(mean=target_input_shift, std=target_input_scale)

    if not (source_model_ensemble_config.model_ensemble_size > 1 and
            source_model_ensemble_config.average_mode == 'probs'):
        source_fmodel = fb.PyTorchModel(source_model, bounds=(0, 1), device=device, preprocessing=source_preprocessing)
    else:
        source_model.to(device)
        source_fmodel = None
    target_model.to(device)


    # Initialize source/target solvers
    source_solvers_solver_options_arr = get_solvers_solver_options_arr(source_solvers_config,
                                                                       source_model_ensemble_config,
                                                                       dtype, device)
    target_solvers_solver_options_arr = get_solvers_solver_options_arr(target_solvers_config,
                                                                       target_model_ensemble_config,
                                                                       dtype, device)
    print(source_solvers_solver_options_arr, target_solvers_solver_options_arr)



    if wandb.config.attack_mode == 'fgsm':

        if source_model_ensemble_config.model_ensemble_size > 1 and source_model_ensemble_config.average_mode == 'probs':
            config_fgsm2ensemble_test = {"eps": None,
                                         "mean": (0.,0.,0.), "std": (1.,1.,1.),
                                         "source_shift": source_input_shift, "source_scale": source_input_scale,
                                         "target_shift": (0.,0.,0.), "target_scale": (1.,1.,1.)}
            attack = ma.FGSM2Ensemble(None,
                                      **config_fgsm2ensemble_test,
                                      average_mode=source_model_ensemble_config.average_mode)
        else:
            attack = fb.attacks.FGSM(random_start=False)

    elif wandb.config.attack_mode == 'pgd':
        attack = fb.attacks.LinfPGD(steps=wandb.config.pgd_niters,
                                    random_start=False,
                                    abs_stepsize=wandb.config.pgd_lr)

    elif wandb.config.attack_mode == 'deepfool':
        attack = fb.attacks.LinfDeepFoolAttack()


    robust_accuracies = attack_validate(attack,
                                        epsilons,
                                        test_loader,
                                        source_model_ensemble_config,
                                        source_solvers_solver_options_arr,
                                        target_solvers_solver_options_arr,
                                        source_model=source_model,
                                        source_fmodel=source_fmodel,
                                        target_model=target_model,
                                        data_inv_normalize=data_inv_normalize,
                                        target_normalize=target_normalize,)
    for eps, acc in zip(epsilons, robust_accuracies):
        wandb.log({'eps': int(eps*255), 'robust_accuracy': acc})

