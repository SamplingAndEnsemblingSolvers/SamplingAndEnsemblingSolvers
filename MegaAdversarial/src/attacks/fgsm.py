from .attack import Attack, Attack2Ensemble
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import itertools



class FGSM(Attack):
    """
    The standard FGSM attack. Assumes (0, 1) normalization.
    """

    def __init__(self, model, eps=None,
                 mean=None, std=None,
                 source_shift=None, source_scale=None,
                 target_shift=None, target_scale=None,):
        super(FGSM, self).__init__(model)
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.mean = mean if mean is not None else (0., 0., 0.)
        self.std = std if std is not None else (1., 1., 1.)
        
        self.source_shift = source_shift if source_shift is not None else (0., 0., 0.)
        self.source_scale = source_scale if source_scale is not None else (1., 1., 1.)
        
        self.target_shift = target_shift if target_shift is not None else (0., 0., 0.)
        self.target_scale = target_scale if target_scale is not None else (1., 1., 1.)

    def forward(self, x, y, kwargs):

        training = self.model.training
        if training:
            self.model.eval()
            
        # range of the image from dataset -> [0, 1]
        data_inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std]) 

        # [0, 1] -> range for the forward of source model
        source_normalize = transforms.Normalize(mean=self.source_shift, std=self.source_scale)
        
        # [0, 1] -> range for the forward of target model
        target_normalize = transforms.Normalize(mean=self.target_shift, std=self.target_scale) 

        x = data_inv_normalize(x) # x in [0, 1]

        x_attacked = x.clone().detach()
        x_attacked.requires_grad_(True)
        loss = self.loss_fn(self.model(source_normalize(x_attacked), **kwargs), y)
        grad = torch.autograd.grad(
            [loss], [x_attacked], create_graph=False, retain_graph=False
        )[0]
        x_attacked = x_attacked + self.eps * grad.sign()
        x_attacked = self._project(x_attacked)
        x_attacked = x_attacked.detach()
        x_attacked = target_normalize(x_attacked)
        if training:
            self.model.train()
        return x_attacked, y


def clamp(X, lower_limit, upper_limit):
    if not isinstance(upper_limit, torch.Tensor):
        upper_limit = torch.tensor(upper_limit, device=X.device, dtype=X.dtype)
    if not isinstance(lower_limit, torch.Tensor):
        lower_limit = torch.tensor(lower_limit, device=X.device, dtype=X.dtype)
    return torch.max(torch.min(X, upper_limit), lower_limit)


class FGSMRandom(Attack):
    """
    The standard FGSM attack. Assumes (0, 1) normalization.
    This implementation is inspired by the implementation from here:
    https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py
    """

    def __init__(self, model, alpha, epsilon=None, mu=None, std=None):
        '''
        Args:
            model: the neural network model
            alpha: the step size
            epsilon: the radius of the random noise
            mu: the mean value of all dataset samples
            std: the std value of all dataset samples
        '''
        super(FGSMRandom, self).__init__(model)
        self.epsilon = epsilon
        self.alpha = alpha
        if (mu is not None) and (std is not None):
            mu = torch.tensor(mu, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

            self.lower_limit = (0. - mu) / std
            self.upper_limit = (1. - mu) / std # lower = -mu/std, upper=(1-mu)/std

            self.epsilon = self.epsilon / std
            self.alpha = self.alpha / std
        else:
            self.lower_limit = 0.
            self.upper_limit = 1.

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def forward(self, x, y, kwargs):
        training = self.model.training
        if training:
            self.model.eval()

        delta = self.epsilon - (2 * self.epsilon) * torch.rand_like(x)  # Uniform[-eps, eps]
        delta.data = clamp(delta, self.lower_limit - x, self.upper_limit - x)
        delta.requires_grad = True
        output = self.model(x + delta, **kwargs)
        loss = self.loss_fn(output, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = clamp(delta, self.lower_limit - x, self.upper_limit - x)
        delta = delta.detach()

        if training:
            self.model.train()
        return x + delta, y


class FGSM2Ensemble(Attack2Ensemble):
    """
    The standard FGSM attack. Assumes (0, 1) normalization.
    """

    def __init__(self, models, eps=None,
                 mean=None, std=None,
                 source_shift=None, source_scale=None,
                 target_shift=None, target_scale=None,
                 average_mode='probs',):
        '''
            average_mode: str
                'probs' if we compute grad = dloss(sum(prob_i)/n)/dx, x_attacked = x + eps * grad.sign();
                'perturbs' if we compute grad_i = dloss(prob_i)/dx, x_attacked_i = x + eps * grad_i.sign(), x_attacked=sum(x_attacked_i)/n, x_attacked = clamp(x_attacked, x-eps, x+eps)
        '''
        super(FGSM2Ensemble, self).__init__(models)
        self.eps = eps
        self.loss_fn = nn.NLLLoss().to(self.device)
        self.mean = mean if mean is not None else (0., 0., 0.)
        self.std = std if std is not None else (1., 1., 1.)
        
        self.average_mode = average_mode
        
        self.source_shift = source_shift if source_shift is not None else (0., 0., 0.)
        self.source_scale = source_scale if source_scale is not None else (1., 1., 1.)
        
        self.target_shift = target_shift if target_shift is not None else (0., 0., 0.)
        self.target_scale = target_scale if target_scale is not None else (1., 1., 1.)

    def forward(self, x, y, kwargs_arr):

        training = self.models[0].training
        if training:
            for model in self.models:
                model.eval()

        data_inv_normalize = transforms.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)], std=[1/s for s in self.std]) # range of the image from dataset -> [0, 1]

        source_normalize = transforms.Normalize(mean=self.source_shift, std=self.source_scale) # [0, 1] -> range for the forward of source model
        target_normalize = transforms.Normalize(mean=self.target_shift, std=self.target_scale) # [0, 1] -> range for the forward of target model
        
        x = data_inv_normalize(x) # x in [0, 1]
        
        if self.average_mode == 'probs':
            probs_ensemble = 0
            x_attacked = x.clone().detach()
            x_attacked.requires_grad_(True)
            
            for n, (model, kwargs) in enumerate(itertools.zip_longest(self.models, kwargs_arr, fillvalue=self.models[0])): 
                logits = model(source_normalize(x_attacked), **kwargs)
                probs_ensemble = probs_ensemble + nn.Softmax(dim=-1)(logits)
            probs_ensemble /= (n + 1)

            loss = self.loss_fn(torch.log(probs_ensemble), y)
            grad = torch.autograd.grad(
                [loss], [x_attacked], create_graph=False, retain_graph=False
            )[0]
            
            x_attacked = x_attacked + self.eps * grad.sign()
            x_attacked = self._project(x_attacked)
            x_attacked = x_attacked.detach()
            
        elif self.average_mode == 'perturbs':
            x_attacked = 0
            
            for n, (model, kwargs) in enumerate(itertools.zip_longest(self.models, kwargs_arr, fillvalue=self.models[0])): 
                x_attacked_n = x.clone().detach()
                x_attacked_n.requires_grad_(True)
                
                logits = model(source_normalize(x_attacked_n), **kwargs)
                loss = self.loss_fn(torch.log(nn.Softmax(dim=-1)(logits)), y)
                grad = torch.autograd.grad(
                    [loss], [x_attacked_n], create_graph=False, retain_graph=False
                )[0]
                
                x_attacked_n = x_attacked_n + self.eps * grad.sign()
                x_attacked =  x_attacked + x_attacked_n.detach()
            x_attacked = x_attacked / (n + 1)
            
            x_attacked = self._clamp(x_attacked, x - self.eps, x + self.eps)
            x_attacked = self._project(x_attacked)
            x_attacked = x_attacked.detach()
        
        x_attacked = target_normalize(x_attacked)
        
        if training:
            for model in self.models:
                model.train()
        return x_attacked, y

    
class FGSMRandom2Ensemble(Attack2Ensemble):
    """
    The standard FGSM attack. Assumes (0, 1) normalization.
    This implementation is inspired by the implementation from here:
    https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py
    """

    def __init__(self, models, alpha, epsilon=None, mu=None, std=None, average_mode='probs'):
        '''
        Args:
            models: list of neural network models
            alpha: the step size
            epsilon: the radius of the random noise
            mu: the mean value of all dataset samples
            std: the std value of all dataset samples
        '''
        super(FGSMRandom2Ensemble, self).__init__(models)
        self.epsilon = epsilon
        self.alpha = alpha
        if (mu is not None) and (std is not None):
            mu = torch.tensor(mu, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

            self.lower_limit = (0. - mu) / std
            self.upper_limit = (1. - mu) / std # lower = -mu/std, upper=(1-mu)/std

            self.epsilon = self.epsilon / std
            self.alpha = self.alpha / std
        else:
            self.lower_limit = 0.
            self.upper_limit = 1.

#         self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.loss_fn = nn.NLLLoss().to(self.device)
        self.average_mode = average_mode
        

    def forward(self, x, y, kwargs_arr):
        
        if self.average_mode != 'probs':
            raise NotImplementedError

        training = self.models[0].training
        if training:
            for model in self.models:
                model.eval()
            
        delta = self.epsilon - (2 * self.epsilon) * torch.rand_like(x)  # Uniform[-eps, eps]
        delta.data = clamp(delta, self.lower_limit - x, self.upper_limit - x)
        delta.requires_grad = True
        
        probs_ensemble = 0
        
        # Here we compute probs using many solvers
        for n, (model, kwargs) in enumerate(itertools.zip_longest(self.models, kwargs_arr, fillvalue=self.models[0])): 
            logits = model(x + delta, **kwargs)
            probs_ensemble = probs_ensemble + nn.Softmax(dim=-1)(logits)
        probs_ensemble /= (n + 1)
        output = torch.log(probs_ensemble)

        loss = self.loss_fn(output, y)
        
        loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = clamp(delta, self.lower_limit - x, self.upper_limit - x)
        delta = delta.detach()

        if training:
            for model in self.models:
                model.train()
        return x + delta, y