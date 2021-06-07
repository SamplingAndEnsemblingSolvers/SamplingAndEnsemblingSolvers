import torch
from collections import namedtuple
import abc

from .misc import _compute_error_ratio, _optimal_step_size, _select_initial_step, _interp_fit, _interp_evaluate

_RungeKuttaState = namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')

class RKAdaptiveStepsizeSolver(object, metaclass = abc.ABCMeta):
    def __init__(self,
                 dtype=None, device=None,
                 rtol=None, atol=None, norm=None,
                 first_step=None, max_num_steps=2**31-1,
                 safety=0.9, ifactor=10., dfactor=0.2,
                ):
        self.dtype = dtype
        self.device = device
        
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.norm = norm
        
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        
        self.n_steps_made = 0
        
        
    def _get_ButcherTableau(self):
        return {'c': self.c, 'w': self.w, 'b': self.b, 'b_error': self.b_error}
    
    
    @abc.abstractmethod
    def _make_step(self, rhs_func, y0, f0, t, dt):
        pass
    
    
    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.type_as(y0)
#         y_mid = y0 + k.matmul(dt * self.mid).view_as(y0)
        y_mid = y0 + torch.einsum('tbchw,t->bchw', torch.stack(k), dt*self.mid)
        f0 = k[0]
        f1 = k[-1]
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)
    
    
    def _update_rk_state(self, rhs_func, rk_state):
        y0, f0, _, t0, dt, interp_coeff = rk_state
        
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(y0).all(), 'non-finite values in state `y`: {}'.format(y0)
        
        y1, f1, y1_error, k = self._make_step(rhs_func, y0, f0, t0, dt)
        
        error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
        accept_step = error_ratio <= 1
        
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        
        interp_coeff = self._interp_fit(y0, y1, k, dt) if accept_step else interp_coeff
        
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state
        
    
    def _make_adaptive_step(self, rhs_func, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._update_rk_state(rhs_func, self.rk_state)
            n_steps += 1
            
        print('n_steps', n_steps, flush=True)
        self.n_steps_made += n_steps
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)
    
    
    def integrate(self, rhs_func, x, t):
        t = t.to(dtype=self.dtype, device=self.device)
        f0 = rhs_func(t[0], x) # y0 = x
        
        ## Estimate size of the first step
        if self.first_step is None:
            first_step = _select_initial_step(rhs_func, t[0], x, self.order - 1,
                                              self.rtol, self.atol, self.norm, f0=f0)
        else:
            first_step = self.first_step
            
#         print(first_step, flush=True)
        self.rk_state = _RungeKuttaState(x, f0, t[0], t[0], first_step, [x] * 5)
        
        ## Integrate
        solution = torch.empty(len(t), *x.shape, dtype=x.dtype, device=x.device)
        solution[0] = x
        
        for i in range(1, len(t)):
            print(i, flush=True)
            solution[i] = self._make_adaptive_step(rhs_func, t[i])
        
        return solution
    
    
    @property
    @abc.abstractmethod
    def order(self):
        pass
    