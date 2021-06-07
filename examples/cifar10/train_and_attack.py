import os
import argparse
from argparse import Namespace
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm, weight_norm
from functools import partial
import itertools
import random

from decimal import Decimal
import wandb
import sys

sys.path.append('../../../')
from sopa.src.solvers.utils import create_solver, noise_params
from sopa.src.models.utils import fix_seeds, RunningAverageMeter

import sopa.src.models.odenet_cifar10.layers as cifar10_models
from sopa.src.models.odenet_cifar10.layers import MetaLayer 
from sopa.src.models.odenet_cifar10.data import inf_generator, get_cifar10_train_val_loaders, get_cifar10_test_loader
from sopa.src.models.odenet_cifar10.utils import *

from sopa.src.models.odenet_mnist.utils import makedirs, learning_rate_with_decay
from MegaAdversarial.src.attacks import (
    Clean,
    PGD,
    FGSM,
    FGSMRandom,
    Clean2Ensemble,
    FGSM2Ensemble,
    FGSMRandom2Ensemble,
)
import apex.amp as amp

parser = argparse.ArgumentParser()
# Architecture params
parser.add_argument('--is_odenet', type=eval, default=True, choices=[True, False])
parser.add_argument('--network', type=str, choices=['metanode34', 'metanode18', 'metanode10', 'metanode6', 'metanode4',
                                                    'premetanode34', 'premetanode18', 'premetanode10', 'premetanode6',
                                                    'premetanode4'],
                    default='premetanode10')
parser.add_argument('--in_planes', type=int, default=64)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--checkpoint_wnb_run_path', type=str, default=None)
parser.add_argument('--checkpoint_wnb_name', type=str, default=None)


# Solver params
parser.add_argument('--solvers',
                    type=lambda s: [tuple(map(lambda iparam: str(iparam[1]) if iparam[0] <= 1 else (
                        int(iparam[1]) if iparam[0] == 2 else (
                            float(iparam[1]) if iparam[0] == 3 else Decimal(iparam[1]))),
                                              enumerate(item.split(',')))) for item in s.strip().split(';')],
                    default="rk2,u,2,-1,0.3,-1",
                    help='Each solver is represented with (method,parameterization,n_steps,step_size,u0,v0) \n' +
                         'If the solver has only one parameter u0, set v0 to -1; \n' +
                         'n_steps and step_size are exclusive parameters, only one of them can be != -1, \n'
                         'If n_steps = step_size = -1, automatic time grid_constructor is used \n;'
                         'For example, --solvers rk4,uv,2,-1,0.3,0.6;rk3,uv,-1,0.1,0.4,0.6;rk2,u,4,-1,0.3,-1')

parser.add_argument('--solver_mode', type=str, choices=['switch', 'ensemble', 'standalone'], default='standalone')
parser.add_argument('--val_solver_modes',
                    type=lambda s: s.strip().split(','),
                    default=['standalone'],
                    help='Solver modes to use for validation step')

# Params for MetaSolver in ewitch mode
parser.add_argument('--switch_probs', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help="--switch_probs 0.8,0.1,0.1")

# Params for MetaSolver in ensemble(averaging ODE outputs) mode
parser.add_argument('--ensemble_weights', type=lambda s: [float(item) for item in s.split(',')], default=None,
                    help="ensemble_weights 0.6,0.2,0.2")
parser.add_argument('--ensemble_prob', type=float, default=1.)

# Params for MetaNode in ensemble(averaging probabilities) mode
parser.add_argument('--model_ensemble_size', type=int, default=1, help='Number of models in the ensemble')
parser.add_argument('--model_ensemble_average_mode', type=int, default=None, help='How to average models during training', choices=['probs'])

# Noise params that are added to solver params u/v
parser.add_argument('--noise_type', type=str, choices=['cauchy', 'normal'], default=None)
parser.add_argument('--noise_sigma', type=float, default=0.001)
parser.add_argument('--noise_prob', type=float, default=0.)

# Set u=2/3 for RK2?
parser.add_argument('--minimize_rk2_error', type=eval, default=False, choices=[True, False])

# Training params
parser.add_argument('--seed', type=int, default=502)
parser.add_argument('--nepochs_nn', type=int, default=500)
parser.add_argument('--nepochs_solver', type=int, default=0)
parser.add_argument('--nstages', type=int, default=1)

# Loaders params
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--val_perc', type=float, default=0.1,
                    help='Percentage split of the training set used for the validation set. \n'
                         'Should be a float in the range [0, 1].')
parser.add_argument('--gpu', type=int, default=0)

# Optimizer params
parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'rmsprop', 'adam'])
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_uv', type=float, default=1e-3, help='lr, when SGD/Adam are used for solver params training')
parser.add_argument('--zero_grad_every', type=int, default=1)


# LR scheduler params
parser.add_argument('--base_lr', type=float, default=1e-5, help='base_lr for CyclicLR scheduler \n' +
                                                                'or SGD/Adam initial lr')
parser.add_argument('--max_lr', type=float, default=None, help='max_lr for CyclicLR scheduler')
parser.add_argument('--step_size_up', type=int, default=6000, help='step_size_up for CyclicLR scheduler')
parser.add_argument('--cyclic_lr_mode', type=str, default='triangular2', help='mode for CyclicLR scheduler')
parser.add_argument('--grad_clipping_threshold', type=float, default=None, help='Threshold for gradients clipping')

parser.add_argument('--init', type=str, default=None, help='Weight initialization, might be None or orthogonal')

# Params for Mixed-precision training
parser.add_argument('--opt-level', default='O0', type=str, choices=['O0', 'O1', 'O2'],
                    help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
                    help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
parser.add_argument('--master-weights', type=eval, default=False, choices=[True, False],
                    help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
parser.add_argument('--torch_dtype', type=str, default='float32')
parser.add_argument('--wandb_project_name', type=str, default=None)

parser.add_argument('--data_root', type=str, default='./data')  # Not in sweep config
parser.add_argument('--save_dir', type=str, default='./')  # Not in sweep config
parser.add_argument('--debug', action='store_true')  # Not in sweep config

# Noise and adversarial attacks parameters:
parser.add_argument('--data_noise_std', type=float, default=0.,
                    help='Applies Norm(0, std) gaussian noise to each training batch')
parser.add_argument('--eps_adv_training', type=float, default=0.3,
                    help='Epsilon for adversarial training')
parser.add_argument(
    "--adv_training_mode",
    default="clean",
    choices=["clean", "fgsm", "at", "fgsm_random"],
    help='Adverarial training method/mode, by default there is no adversarial training (clean). \n'
         '"fgsm" and "at" correspond the training with FGSM and PGD attacks correspondingly. \n'
         '"fgsm_random" corresponds to the adversarial training described in https://arxiv.org/abs/2001.03994')
parser.add_argument('--fgsm_random_step_size_training', type=float, default=0.0392156862745098,
                    help='FGSMRandom step size, default=10/255')
parser.add_argument('--pgd_lr_training', type=float, default=2. / 255,
                    help='lr of PGD for training')
parser.add_argument('--pgd_niter_training', type=int, default=7,
                    help='Number of PGD iterations for training')
parser.add_argument('--ss_loss', type=eval, default=False, choices=[True, False])
parser.add_argument('--ss_loss_reg', type=float, default=0.1)

parser.add_argument('--eps_adv_testing', type=float, default=0.3,
                    help='Epsilon for adversarial testing')
parser.add_argument('--adv_testing_mode',
                    default="clean",
                    choices=["clean", "fgsm", "at"],
                    help='''Adversarsarial testing mode''')
parser.add_argument('--pgd_lr_testing', type=float, default=2. / 255,
                    help='lr of PGD for testing')
parser.add_argument('--pgd_niter_testing', type=int, default=7,
                    help='Number of PGD iterations for testing')

# Type of layer's output normalization
parser.add_argument('--normalization_resblock', type=str, default='BN',
                    choices=['BN', 'GN', 'LN', 'IN', 'NF'])
parser.add_argument('--normalization_odeblock', type=str, default='NF',
                    choices=['BN', 'GN', 'LN', 'IN', 'NF'])
parser.add_argument('--normalization_bn1', type=str, default='BN',
                    choices=['BN', 'GN', 'LN', 'IN', 'NF'])
parser.add_argument('--num_gn_groups', type=int, default=32, help='Number of groups for GN normalization')

# Type of layer's weights  normalization
parser.add_argument('--param_normalization_resblock', type=str, default='PNF',
                    choices=['WN', 'SN', 'PNF'])
parser.add_argument('--param_normalization_odeblock', type=str, default='WN',
                    choices=['WN', 'SN', 'PNF'])
parser.add_argument('--param_normalization_bn1', type=str, default='PNF',
                    choices=['WN', 'SN', 'PNF'])
# Type of activation
parser.add_argument('--activation_resblock', type=str, default='ReLU',
                    choices=['ReLU', 'GeLU', 'Softsign', 'Tanh', 'AF'])
parser.add_argument('--activation_odeblock', type=str, default='ReLU',
                    choices=['ReLU', 'GeLU', 'Softsign', 'Tanh', 'AF'])
parser.add_argument('--activation_bn1', type=str, default='ReLU',
                    choices=['ReLU', 'GeLU', 'Softsign', 'Tanh', 'AF'])

args, unknown_args = parser.parse_known_args()

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader, device, solvers=None, solver_options=None):
    model.eval()
    total_correct = 0

    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)

        with torch.no_grad():
            if solvers is not None:
                out = model(x, solvers, solver_options).cpu().detach().numpy()
            else:
                out = model(x).cpu().detach().numpy()
            predicted_class = np.argmax(out, axis=1)
            total_correct += np.sum(predicted_class == target_class)

    total = len(dataset_loader) * dataset_loader.batch_size

    return total_correct / total


def adversarial_accuracy(model, dataset_loader, device, solvers=None, solver_options=None, args=None):
    global CONFIG_PGD_TEST
    global CONFIG_FGSM_TEST

    model.eval()
    total_correct = 0

    if args.adv_testing_mode == "clean":
        test_attack = Clean(model)
    elif args.adv_testing_mode == "fgsm":
        test_attack = FGSM(model, **CONFIG_FGSM_TEST)
    elif args.adv_testing_mode == "at":
        test_attack = PGD(model, **CONFIG_PGD_TEST)
    else:
        raise ValueError("Attack type not understood.")
    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        x, y = test_attack(x, y, {"solvers": solvers, "solver_options": solver_options})
        y = one_hot(np.array(y.cpu().numpy()), 10)
        target_class = np.argmax(y, axis=1)

        with torch.no_grad():
            if solvers is not None:
                out = model(x, solvers, solver_options).cpu().detach().numpy()
            else:
                out = model(x).cpu().detach().numpy()
            predicted_class = np.argmax(out, axis=1)
            total_correct += np.sum(predicted_class == target_class)

    total = len(dataset_loader) * dataset_loader.batch_size

    return total_correct / total


def accuracy_ensemble(models, dataset_loader, device, solvers_solver_options_arr=None, data_noise_std=None):
    for model in models:
        model.eval()
    total_correct = 0

    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)

        with torch.no_grad():
            # Add noise:
            if (data_noise_std is not None) and (data_noise_std > 1e-12):
                x = x + data_noise_std * torch.randn_like(x)

            probs_ensemble = 0

            if solvers_solver_options_arr is not None:
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

            predicted_class = np.argmax(probs_ensemble, axis=1)
            total_correct += np.sum(predicted_class == target_class)

    total = len(dataset_loader) * dataset_loader.batch_size
    return total_correct / total


def adversarial_accuracy_ensemble(models, dataset_loader, device, solvers_solver_options_arr=None, args=None):
    global CONFIG_PGD_TEST
    global CONFIG_FGSM_TEST

    for model in models:
        model.eval()
    total_correct = 0

    if args.a_adv_testing_mode == "clean":
        test_attack = Clean2Ensemble(models)
    elif args.a_adv_testing_mode == "fgsm":
        test_attack = FGSM2Ensemble(models, average_mode=args.model_ensemble_average_mode, **CONFIG_FGSM_TEST)
    else:
        raise ValueError("Attack type is not implemented for ensemble of models")

    for x, y in dataset_loader:
        x, y = x.to(device), y.to(device)
        x, y = test_attack(x, y, solvers_solver_options_arr)
        y = one_hot(np.array(y.cpu().numpy()), 10)
        target_class = np.argmax(y, axis=1)

        with torch.no_grad():

            probs_ensemble = 0

            if solvers_solver_options_arr is not None:
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

            predicted_class = np.argmax(probs_ensemble, axis=1)
            total_correct += np.sum(predicted_class == target_class)

    total = len(dataset_loader) * dataset_loader.batch_size
    return total_correct /total


def train(model,
          data_gen,
          solvers,
          solver_options,
          criterion,
          optimizer,
          device,
          is_odenet=True,
          iter=None,
          args=None):
    model.train()

    if (iter + 1) % args.zero_grad_every == 0:
        optimizer.zero_grad()

    x, y = data_gen.__next__()
    x = x.to(device)
    y = y.to(device)

    ### Noise params
    if args.noise_type is not None:
        for i in range(len(solvers)):
            solvers[i].u, solvers[i].v = noise_params(solvers[i].u0,
                                                      solvers[i].v0,
                                                      std=args.noise_sigma,
                                                      bernoulli_p=args.noise_prob,
                                                      noise_type=args.noise_type)
            solvers[i].build_ButcherTableau()
            
    global CONFIG_PGD_TRAIN
    global CONFIG_FGSM_TRAIN
    global CONFIG_FGSMRandom_TRAIN

    if args.model_ensemble_size == 1:
    
        if args.adv_training_mode == "clean":
            train_attack = Clean(model)
        elif args.adv_training_mode == "fgsm":
            train_attack = FGSM(model, **CONFIG_FGSM_TRAIN)
        elif args.adv_training_mode == "fgsm_random":
            train_attack = FGSMRandom(model, **CONFIG_FGSMRandom_TRAIN)
        elif args.adv_training_mode == "at":
            train_attack = PGD(model, **CONFIG_PGD_TRAIN)
        else:
            raise ValueError("Attack type not understood.")
            
    elif args.model_ensemble_size > 1:
        if args.adv_training_mode == "clean":
            train_attack = Clean2Ensemble([model])
        elif args.adv_training_mode == "fgsm":
            train_attack = FGSM2Ensemble([model], average_mode=args.model_ensemble_average_mode, **CONFIG_FGSM_TRAIN)
        elif args.adv_training_mode == "fgsm_random":
            train_attack = FGSMRandom2Ensemble([model], average_mode=args.model_ensemble_average_mode, **CONFIG_FGSMRandom_TRAIN)
        else:
            raise ValueError("Attack type not understood.")
        
    if args.model_ensemble_size == 1:
        x, y = train_attack(x, y, {"solvers": solvers, "solver_options": solver_options})
    elif args.model_ensemble_size > 1:
        solvers_solver_options_arr = [{'solvers': [solver], 'solver_options': solver_options} for solver in solvers]
        x, y = train_attack(x, y, solvers_solver_options_arr)

    ### Add noise:
    if args.data_noise_std > 1e-12:
        with torch.no_grad():
            x = x + args.data_noise_std * torch.randn_like(x)
    
    ### Forward pass
    if args.model_ensemble_size == 1:
        if is_odenet:
            logits = model(x, solvers, solver_options, Namespace(ss_loss=args.ss_loss))
        else:
            logits = model(x)

        xentropy = criterion(logits, y)
        
        ss_loss = 0.
        if args.ss_loss:
            for m in model.named_modules():
                if isinstance(m, MetaLayer):
                    ss_loss = ss_loss + m.get_ss_loss()
            loss = xentropy + args.ss_loss_reg * ss_loss
        else:
            ss_loss = 0.
            loss = xentropy

    elif args.model_ensemble_size > 1:
        if not is_odenet:
            raise NotImplementedError
            
        if not (args.model_ensemble_average_mode == 'probs'):
            raise NotImplementedError
            
        probs_ensemble = 0
        for n, (model, kwargs) in enumerate(itertools.zip_longest([model], solvers_solver_options_arr, fillvalue=model)): 
            logits = model(x, **kwargs)
            probs_ensemble = probs_ensemble + nn.Softmax(dim=-1)(logits)
        probs_ensemble /= (n + 1)
        
        xentropy = criterion(torch.log(probs_ensemble), y)

        if args.ss_loss:
            raise NotImplementedError 
        ss_loss = 0
        loss = xentropy
            
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()

    if args.grad_clipping_threshold:
        grad_norm = nn.utils.clip_grad_norm(amp.master_params(optimizer), args.grad_clipping_threshold)

    if (iter + 1) % args.zero_grad_every == 0:
        optimizer.step()

    ### Denoise params
    if args.noise_type is not None:
        for i in range(len(solvers)):
            solvers[i].u, solvers[i].v = solvers[i].u0, solvers[i].v0
            solvers[i].build_ButcherTableau()

    if args.ss_loss:
        return {'xentropy': xentropy.item(), 'ss_loss': ss_loss}
    return {'xentropy': xentropy.item()}


def update_solvers_with_sweep_params(solvers, wandb):
    '''Update solver parameters with parameters set by wandb sweep

    :param solvers: list of lists
        Example: [['rk4','uv','2','-1','0.3','0.6']]
    :param wandb: WnB params
    :return: list of lists are updated inplace
    '''
    for solver in solvers:
        if ('wnb_method' in wandb.config) and (wandb.config.wnb_method is not None):
            solver[0] = wandb.config.wnb_method
        if ('wnb_parameterization' in wandb.config) and (wandb.config.wnb_parameterization is not None):
            solver[1] = wandb.config.wnb_parameterization
        if ('wnb_n_steps' in wandb.config) and (wandb.config.wnb_n_steps is not None):
            solver[2] = wandb.config.wnb_n_steps
        if ('wnb_u' in wandb.config) and (wandb.config.wnb_u is not None):
            solver[-2] = wandb.config.wnb_u
        if ('wnb_v' in wandb.config) and (wandb.config.wnb_v is not None):
            solver[-1] = wandb.config.wnb_v

def set_max_base_lr(wandb):
    if wandb.config.max_lr is not None and wandb.config.base_lr is not None:
        max_lr, base_lr = wandb.config.max_lr, wandb.config.base_lr
    elif wandb.config.max_lr is not None:
        max_lr = wandb.config.max_lr
        base_lr = wandb.config.max_lr / wandb.config.max_lr_reduction
    elif wandb.config.base_lr is not None:
        base_lr = wandb.config.base_lr
        max_lr = base_lr
    else:
        raise ValueError('Either max_lr or base_lr should be defined in WnB config')
    return max_lr, base_lr

if __name__ == "__main__":
    wandb.init(project="sampling-ensembling-solvers-node", anonymous="allow",)
    wandb.config.update(args)

    fix_seeds(wandb.config.seed)

    # Path to save checkpoints locally in <args.save_dir>/<entity>/<project>/<run_id> [Julia style]
    makedirs(wandb.config.save_dir)
    makedirs(os.path.join(wandb.config.save_dir, wandb.run.path))

    if wandb.config.torch_dtype == 'float64':
        dtype = torch.float64
    elif wandb.config.torch_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('torch_type should be either float64 or float32')

    device = torch.device('cuda:' + str(wandb.config.gpu) if torch.cuda.is_available() else 'cpu')

    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    CONFIG_FGSM_TEST = {"eps": wandb.config.eps_adv_testing,
                        "mean": cifar10_mean, "std": cifar10_std,
                        "source_shift": cifar10_mean, "source_scale": cifar10_std,
                        "target_shift": cifar10_mean, "target_scale": cifar10_std,
                       }
    CONFIG_FGSM_TRAIN = {"eps": wandb.config.eps_adv_training,
                         "mean": cifar10_mean, "std": cifar10_std,
                         "source_shift": cifar10_mean, "source_scale": cifar10_std,
                         "target_shift": cifar10_mean, "target_scale": cifar10_std,
                        }

    CONFIG_PGD_TEST = {"eps": wandb.config.eps_adv_testing, "lr": wandb.config.pgd_lr_testing,
                       "n_iter": wandb.config.pgd_niter_testing,
                       "mean": cifar10_mean,
                       "std": cifar10_std}
    CONFIG_PGD_TRAIN = {"eps": wandb.config.eps_adv_training, "lr": wandb.config.pgd_lr_training,
                        "n_iter": wandb.config.pgd_niter_training,
                        "mean": cifar10_mean,
                        "std": cifar10_std
                        }

    fgsm_random_eps = wandb.config.eps_adv_training
    fgsm_random_alpha = wandb.config.fgsm_random_step_size_training

    CONFIG_FGSMRandom_TRAIN = {"epsilon": fgsm_random_eps,
                               "alpha": fgsm_random_alpha,
                               "mu": cifar10_mean,
                               "std": cifar10_std}

    ### Create train / val solvers
    print(wandb.config.solvers)

    if ';' in wandb.config.solvers:
        splitted_solvers = [tuple(map(lambda iparam: str(iparam[1]) if iparam[0] <= 1 else (
                        int(iparam[1]) if iparam[0] == 2 else (
                            float(iparam[1]) if iparam[0] == 3 else Decimal(iparam[1]))),
                                              enumerate(item.split(',')))) for item in wandb.config.solvers.strip().split(';')]
        train_solvers = [create_solver(*solver_params, dtype=dtype, device=device) for solver_params in splitted_solvers]
    else:
        update_solvers_with_sweep_params(wandb.config.solvers, wandb)
        train_solvers = [create_solver(*solver_params, dtype=dtype, device=device) for solver_params in
                     wandb.config.solvers]

    for solver in train_solvers:
        solver.freeze_params()

    train_solver_options = Namespace(**{key: wandb.config[key] for key in ['solver_mode', 'switch_probs',
                                                                           'ensemble_prob', 'ensemble_weights']})
    val_solver_modes = wandb.config.val_solver_modes

    ### Build the model
    # Initialize normalization layers
    norm_layers = (get_normalization(wandb.config.normalization_resblock, wandb.config.num_gn_groups),
                   get_normalization(wandb.config.normalization_odeblock, wandb.config.num_gn_groups),
                   get_normalization(wandb.config.normalization_bn1, wandb.config.num_gn_groups))

    param_norm_layers = (get_param_normalization(wandb.config.param_normalization_resblock),
                         get_param_normalization(wandb.config.param_normalization_odeblock),
                         get_param_normalization(wandb.config.param_normalization_bn1))

    act_layers = (get_activation(wandb.config.activation_resblock),
                  get_activation(wandb.config.activation_odeblock),
                  get_activation(wandb.config.activation_bn1))

    # Build Neural ODE model
    is_odenet = wandb.config.is_odenet
    model = getattr(cifar10_models, wandb.config.network)(norm_layers, param_norm_layers, act_layers,
                                                          wandb.config.in_planes,
                                                          is_odenet=wandb.config.is_odenet)
    if wandb.config.init == 'orthogonal':
        model.apply(conv_init_orthogonal)
        model.apply(fc_init_orthogonal)
    else:
        model.apply(conv_init)
        
    
    # Load weights from the checkpoint
    if wandb.config.checkpoint:
        model.load_state_dict(torch.load(wandb.config.checkpoint)['model'])
    elif (wandb.config.checkpoint_wnb_run_path is not None) and (wandb.config.checkpoint_wnb_name is not None):
        model.load_state_dict(torch.load(os.path.join(wandb.config.save_dir,
                                             wandb.config.checkpoint_wnb_run_path,
                                             wandb.config.checkpoint_wnb_name))['model'])

    model.to(device)
    if dtype == torch.float64:
        model.double()

    wandb.watch(model, log='all')

    ### Create dataloaders
    train_loader, train_eval_loader = get_cifar10_train_val_loaders(data_aug=wandb.config.data_aug,
                                                                    batch_size=wandb.config.batch_size,
                                                                    val_perc=wandb.config.val_perc,
                                                                    data_root=wandb.config.data_root,
                                                                    num_workers=wandb.config.num_workers,
                                                                    pin_memory=True,
                                                                    shuffle=True,
                                                                    random_seed=wandb.config.seed)
    
    test_loader = get_cifar10_test_loader(batch_size=wandb.config.batch_size,
                                          data_root=wandb.config.data_root,
                                          num_workers=wandb.config.num_workers,
                                          pin_memory=True,
                                          shuffle=False,)
    
    # train_loader, test_loader, train_eval_loader = get_cifar10_loaders(data_aug=wandb.config.data_aug,
    #                                                                    batch_size=wandb.config.batch_size,
    #                                                                    test_batch_size=wandb.config.test_batch_size,
    #                                                                    perc=1.0,
    #                                                                    data_root=wandb.config.data_root,
    #                                                                    num_workers=wandb.config.num_workers)
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    ### Create criterion and optimizer
    if wandb.config.model_ensemble_size == 1:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.NLLLoss().to(device)
    loss_options = Namespace(ss_loss=wandb.config.ss_loss)

    ### We change the learning rate according to the cyclic learning rate schedule
    max_lr, base_lr = set_max_base_lr(wandb)
    wandb.config.update({'max_lr': max_lr,  'base_lr': base_lr})

    if wandb.config.optim == 'sgd':
        optimizer = optim.SGD([{"params": model.parameters(), 'lr': base_lr}, ],
                              lr=base_lr,
                              weight_decay=wandb.config.weight_decay,
                              momentum=wandb.config.momentum
                              )
    elif wandb.config.optim == 'rmsprop':
        optimizer = optim.RMSprop([{"params": model.parameters(), 'lr': base_lr}, ],
                                  lr=base_lr,
                                  weight_decay=wandb.config.weight_decay,
                                  )
    elif wandb.config.optim == 'adam':
        optimizer = optim.Adam([{"params": model.parameters(), 'lr': base_lr}, ],
                               lr=base_lr,
                               weight_decay=wandb.config.weight_decay,
                               )

    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, optimizer = amp.initialize(model, optimizer, **amp_args)


    scheduler = optim.lr_scheduler.CyclicLR(optimizer,
                                            base_lr=base_lr,
                                            max_lr=max_lr,
                                            step_size_up=wandb.config.step_size_up,
                                            mode=wandb.config.cyclic_lr_mode,
                                            cycle_momentum=(wandb.config.optim != "adam"))

    ### Train the model
    for itr in range(wandb.config.nepochs_nn * batches_per_epoch):

        wandb.log({'lr_last': scheduler.get_last_lr()[0]})

        train_loss = train(model,
                           data_gen,
                           solvers=train_solvers,
                           solver_options=train_solver_options,
                           criterion=criterion,
                           optimizer=optimizer,
                           device=device,
                           is_odenet=is_odenet,
                           iter=itr,
                           args=wandb.config)

        if itr % batches_per_epoch == 0:
            if wandb.config.model_ensemble_size == 1:
                val_acc = accuracy(model, train_eval_loader, device, solvers=train_solvers,
                                   solver_options=train_solver_options)
                test_acc = accuracy(model, test_loader, device, solvers=train_solvers, solver_options=train_solver_options)
                
                if wandb.config.adv_testing_mode != "clean":
                    adv_val_acc = adversarial_accuracy(model, train_eval_loader, device,
                                                         solvers=train_solvers, solver_options=train_solver_options,
                                                         args=wandb.config)
                    adv_test_acc = adversarial_accuracy(model, test_loader, device,
                                                        solvers=train_solvers, solver_options=train_solver_options,
                                                        args=wandb.config)

            elif wandb.config.model_ensemble_size > 1:
                solvers_solver_options_arr = [{'solvers': [solver], 'solver_options': train_solver_options}
                                              for solver in train_solvers]
                # Compute accuracy
                val_acc = accuracy_ensemble([model], train_eval_loader, device='cuda',
                                            solvers_solver_options_arr=solvers_solver_options_arr,
                                            data_noise_std=wandb.config.data_noise_std)
                test_acc = accuracy_ensemble([model], test_loader, device='cuda',
                                            solvers_solver_options_arr=solvers_solver_options_arr,
                                            data_noise_std=wandb.config.data_noise_std)
                if wandb.config.adv_testing_mode != "clean":
                    adv_val_acc = adversarial_accuracy_ensemble([model], train_eval_loader, device='cuda',
                                                            solvers_solver_options_arr=solvers_solver_options_arr,
                                                            args=wandb.config)
                    adv_test_acc = adversarial_accuracy_ensemble([model], test_loader, device='cuda',
                                                            solvers_solver_options_arr=solvers_solver_options_arr,
                                                            args=wandb.config)
            
            
            log_params = {'val_acc': val_acc,
              'test_acc': test_acc,
              'train_loss': train_loss['xentropy'],
              'epoch': itr // batches_per_epoch,
              }
            if wandb.config.adv_testing_mode != "clean":
                log_params.update({'test_acc_{}'.format(wandb.config.adv_testing_mode): adv_test_acc,
                                   'val_acc_{}'.format(wandb.config.adv_testing_mode): adv_val_acc})
            
            save_path = os.path.join(wandb.config.save_dir, wandb.run.path, "amp_checkpoint_{}.pth".format(itr))
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'iter': itr,
                'wandb_config': dict(wandb.config),
            }
            torch.save(checkpoint, save_path)
            wandb.save(save_path)

            wandb.log(log_params)
        else:
            wandb.log(log_params, commit=False)

        scheduler.step()