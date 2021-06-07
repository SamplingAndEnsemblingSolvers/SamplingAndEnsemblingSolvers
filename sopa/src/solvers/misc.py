import torch

### Auxiallary functions to compute adaptive step size, borrowed from https://github.com/rtqichen/torchdiffeq/
def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    ''' err/tol
    '''
    error_tol = atol + rtol * torch.max(y0.abs(), y1.abs())
    error_ratio = norm(error_estimate / error_tol)
#     print(torch.quantile(torch.max(y0.abs(), y1.abs()), 0.9), error_ratio)
    return error_ratio


@torch.no_grad()
def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    device = last_step.device
  
    safety = safety.to(device)
    ifactor = ifactor.to(device)
    dfactor = dfactor.to(device)
    
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = error_ratio.type_as(last_step)
    
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor)) # (1/error_ratio)**exp = (tol/err)**exp
    
    return last_step * factor


def _select_initial_step(func, t0, y0, order, rtol, atol, norm, f0=None):
    """Empirically select a good initial step.
    The algorithm is described in [1]_.
    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """
    dtype = y0.dtype
    device = y0.device
    t_dtype = t0.dtype
    t0 = t0.to(dtype)

    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + torch.abs(y0) * rtol

    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=dtype, device=device)
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)

    d2 = norm((f1 - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=dtype, device=device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))

    return torch.min(100 * h0, h1).to(t_dtype)


### Auxiallary functions to interpolate, borrowed from https://github.com/rtqichen/torchdiffeq/
def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.
    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.
    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = 2 * dt * (f1 - f0) - 8 * (y1 + y0) + 16 * y_mid
    b = dt * (5 * f0 - 3 * f1) + 18 * y0 + 14 * y1 - 32 * y_mid
    c = dt * (f1 - 4 * f0) - 11 * y0 - 5 * y1 + 16 * y_mid
    d = dt * f0
    e = y0
    
    return [e, d, c, b, a]


def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.
    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.
    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """
    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = (t - t0) / (t1 - t0)

    total = coefficients[0] + x * coefficients[1]
    x_power = x
    for coefficient in coefficients[2:]:
        x_power = x_power * x
        total = total + x_power * coefficient

    return total