import torch

from .rk_adaptive import RKAdaptiveStepsizeSolver

class EulerHein(RKAdaptiveStepsizeSolver):
    def __init__(self, dtype=None, device=None, **kwargs):
        super(EulerHein, self).__init__(**kwargs)
        self.dtype = dtype
        self.device = device
        
        self.c = torch.tensor([0., 1.],dtype=dtype, device=device)
        self.w = [torch.tensor([0.,],dtype=dtype, device=device)] + [torch.tensor([1., 0.],dtype=dtype, device=device)]
        self.b = torch.tensor([1/2., 1/2.],dtype=dtype, device=device)
        self.b_error = torch.tensor([1 - 1/2., 0 - 1/2.],dtype=dtype, device=device)
        
        self.mid = torch.tensor([0.5, 0.], dtype=dtype, device=device)
        
            
    def _get_t(self, t, dt):
        t0 = t 
        t1 = t + self.c[1] * dt
        return (t0, t1)

    
    def _make_step(self, rhs_func, y0, f0, t, dt):
        t0, t1 = self._get_t(t, dt)

        k1 = f0
        k2 = rhs_func(t1, y0 + k1 * self.w[1][0] * dt)
        
        y1 = y0 + (k1 * self.b[0] + k2 * self.b[1]) * dt
        f1 = k2
        
        y1_error = (k1 * self.b_error[0] + k2 * self.b_error[1]) * dt
        
        return y1, f1, y1_error, (k1, k2)
    
    
    @property
    def order(self):
        return 2