import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import operator
import numpy as np


def jacobian(f, x, eps):
    if x.ndimension() == 2:
        assert x.size(0) == 1
        x = x.squeeze()

    e = Variable(torch.eye(len(x)).type_as(get_data_maybe(x)))
    J = []
    for i in range(len(x)):
        J.append((f(x + eps * e[i]) - f(x - eps * e[i])) / (2.0 * eps))
    J = torch.stack(J).transpose(0, 1)
    return J


def expandParam(X, n_batch, nDim):
    if X.ndimension() in (0, nDim):
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([n_batch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def bdiag(d):
    assert d.ndimension() == 2
    nBatch, sz = d.size()
    dtype = d.type() if not isinstance(d, Variable) else d.data.type()
    D = torch.zeros(nBatch, sz, sz).type(dtype)
    I = torch.eye(sz).repeat(nBatch, 1, 1).type(dtype).byte()
    D[I] = d.view(-1)
    return D


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)


def bquad(x, Q):
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1).squeeze(1)


def eclamp(x, lower, upper):
    # In-place!!
    if type(lower) == type(x):
        assert x.size() == lower.size()

    if type(upper) == type(x):
        assert x.size() == upper.size()

    I = x < lower
    x[I] = lower[I] if not isinstance(lower, float) else lower

    I = x > upper
    x[I] = upper[I] if not isinstance(upper, float) else upper

    return x


def get_data_maybe(x):
    return x if not isinstance(x, Variable) else x.data


_seen_tables = []


def table_log(tag, d):
    # TODO: There's probably a better way to handle formatting here,
    # or a better way altogether to replace this quick hack.
    global _seen_tables

    def print_row(r):
        print("| " + " | ".join(r) + " |")

    if tag not in _seen_tables:
        print_row(map(operator.itemgetter(0), d))
        _seen_tables.append(tag)

    s = []
    for di in d:
        assert len(di) in [2, 3]
        if len(di) == 3:
            e, fmt = di[1:]
            s.append(fmt.format(e))
        else:
            e = di[1]
            s.append(str(e))
    print_row(s)


def get_traj(T, u, x_init, dynamics):
    from .mpc import QuadCost, LinDx  # TODO: This is messy.

    if isinstance(dynamics, LinDx):
        F = get_data_maybe(dynamics.F)
        f = get_data_maybe(dynamics.f)
        if f is not None:
            assert f.shape == F.shape[:3]

    x = [get_data_maybe(x_init)]
    for t in range(T):
        xt = x[t]
        ut = get_data_maybe(u[t])
        if t < T - 1:
            # new_x = f(Variable(xt), Variable(ut)).data
            if isinstance(dynamics, LinDx):
                xut = torch.cat((xt, ut), 1)
                new_x = bmv(F[t], xut)
                if f is not None:
                    new_x += f[t]
            else:
                new_x = dynamics(Variable(xt), Variable(ut), t).data
            x.append(new_x)
    x = torch.stack(x, dim=0)
    return x


def get_cost(T, u, cost, dynamics=None, x_init=None, x=None):
    from .mpc import QuadCost, LinDx  # TODO: This is messy.

    assert x_init is not None or x is not None

    if isinstance(cost, QuadCost):
        C = get_data_maybe(cost.C)
        c = get_data_maybe(cost.c)

    if x is None:
        x = get_traj(T, u, x_init, dynamics)

    objs = []
    for t in range(T):
        xt = x[t]
        ut = u[t]
        xut = torch.cat((xt, ut), 1)
        if isinstance(cost, QuadCost):
            obj = 0.5 * bquad(xut, C[t]) + bdot(xut, c[t])
        else:
            obj = cost(xut)
        objs.append(obj)
    objs = torch.stack(objs, dim=0)
    total_obj = torch.sum(objs, dim=0)
    return total_obj


def detach_maybe(x):
    if x is None:
        return None
    return x if not x.requires_grad else x.detach()


def data_maybe(x):
    if x is None:
        return None
    return x.data


def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1, noise_level=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = noise_level
    return fftnoise(f)


def sample_from_data(ts, data, freq):
    # Calculate the time of each sample
    time_stamps = np.arange(0, len(data) / freq, 1 / freq)

    # Find the two nearest sampling points
    idx = binary_search_left(time_stamps, ts)

    # Handle cases where ts is outside the range of time_stamps
    if idx == 0:
        return data[0]
    if idx >= len(data):
        return data[-1]

    # Perform linear interpolation
    t1, t2 = time_stamps[idx - 1], time_stamps[idx]
    d1, d2 = data[idx - 1], data[idx]
    value = d1 + (d2 - d1) * (ts - t1) / (t2 - t1)

    return value


def binary_search_left(time_stamps, ts):
    left, right = 0, len(time_stamps) - 1
    idx = len(time_stamps)  # Default index if ts is greater than all elements

    while left <= right:
        mid = (left + right) // 2
        if time_stamps[mid] < ts:
            left = mid + 1
        else:
            idx = mid
            right = mid - 1

    return idx


def read_double_column_csv(file_path):
    with open(file_path, "r") as file:
        data = file.readlines()
    # extract the second column
    data = [i.split(",")[1] for i in data]
    # Convert to float
    data = [float(i) for i in data]
    return data
