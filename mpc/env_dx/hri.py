import torch
from torchdiffeq import odeint
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import sin, cos, pi

from mpc.c_funcs.hri_ode_c import func as hri_ode_c  # Import the Cython function

import numpy as np
from mpc import util
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("bmh")


class HRIDx(nn.Module):
    def __init__(self, model_params=None, dt=0.01, n_state=12, n_ctrl=1):
        super().__init__()

        self.max_torque = 24.0  # has to be float
        self.dt = dt
        self.n_state = n_state
        self.n_ctrl = n_ctrl

        self.goal_weights = None
        self.goal_state = None
        self.ctrl_penalty = None

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5
        self.lower = -self.max_torque
        self.upper = self.max_torque

        self.params = model_params
        self.human_u = None

    def update_input(self, human_u):
        self.human_u = human_u

    def update_goal(self, goal_state, goal_weights, ctrl_penalty):
        self.goal_state = goal_state
        self.goal_weights = goal_weights
        self.ctrl_penalty = ctrl_penalty

    def forward(self, x, u, t=0):
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 12
        assert u.shape[1] == 1
        assert u.ndimension() == 2

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        u = torch.clamp(u, -self.max_torque, self.max_torque)[:, 0]

        q1 = x[0, 0]
        dq1 = x[0, 1]
        h_q2 = x[0, 2]
        h_dq2 = x[0, 3]
        r_d2 = x[0, 4]
        r_dd2 = x[0, 5]
        r_d3 = x[0, 6]
        r_dd3 = x[0, 7]
        r_q4 = x[0, 8]
        r_dq4 = x[0, 9]
        r_q5 = x[0, 10]
        r_dq5 = x[0, 11]

        tau_1 = 0
        tau_3 = 0
        tau_4 = u  # robot torque

        if self.human_u is not None:
            tau_2 = self.human_u[t]  # human torque
        else:
            tau_2 = 0

        # fmt: off
        (
            dq1, ddq1,
            h_dq2, h_ddq2,
            r_dd2, r_ddd2,
            r_dd3, r_ddd3,
            r_dq4, r_ddq4,
            r_dq5, r_ddq5,
        ) = hri_ode_c(
            self.params,
            q1, dq1,
            h_q2, h_dq2,
            r_d2, r_dd2,
            r_d3, r_dd3,
            r_q4, r_dq4,
            r_q5, r_dq5,
            tau_1,
            tau_2,
            tau_3,
            tau_4,
        )
        # fmt: on

        new_q1 = q1 + self.dt * dq1
        new_dq1 = dq1 + self.dt * ddq1
        new_h_q2 = h_q2 + self.dt * h_dq2
        new_h_dq2 = h_dq2 + self.dt * h_ddq2
        new_r_d2 = r_d2 + self.dt * r_dd2
        new_r_dd2 = r_dd2 + self.dt * r_ddd2
        new_r_d3 = r_d3 + self.dt * r_dd3
        new_r_dd3 = r_dd3 + self.dt * r_ddd3
        new_r_q4 = r_q4 + self.dt * r_dq4
        new_r_dq4 = r_dq4 + self.dt * r_ddq4
        new_r_q5 = r_q5 + self.dt * r_dq5
        new_r_dq5 = r_dq5 + self.dt * r_ddq5

        state = torch.stack(
            (
                torch.tensor([new_q1]),
                torch.tensor([new_dq1]),
                torch.tensor([new_h_q2]),
                torch.tensor([new_h_dq2]),
                torch.tensor([new_r_d2]),
                torch.tensor([new_r_dd2]),
                torch.tensor([new_r_d3]),
                torch.tensor([new_r_dd3]),
                torch.tensor([new_r_q4]),
                torch.tensor([new_r_dq4]),
                torch.tensor([new_r_q5]),
                torch.tensor([new_r_dq5]),
            ),
            dim=1,
        )

        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x, ax=None):
        x = util.get_data_maybe(x.view(-1))
        assert len(x) == 10
        l = self.params[2].item()

        cos_th, sin_th, dth = torch.unbind(x)
        th = np.arctan2(sin_th, cos_th)
        x = sin_th * l
        y = cos_th * l

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.get_figure()

        ax.plot((0, x), (0, y), color="k")
        ax.set_xlim((-l * 1.2, l * 1.2))
        ax.set_ylim((-l * 1.2, l * 1.2))
        return fig, ax

    def get_true_obj(self):
        assert self.goal_state is not None
        assert self.goal_weights is not None
        assert self.ctrl_penalty is not None

        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.n_ctrl)))
        assert not hasattr(self, "mpc_lin")
        px = -torch.sqrt(self.goal_weights) * self.goal_state  # + self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


class HRIDx_Sim(nn.Module):
    def __init__(self, model_params=None, u=None, current_t=0.0, dt=None):
        super().__init__()

        self.params = model_params
        self.u = u
        self.current_t = current_t
        self.dx = dt
        self.freq = 1 / dt

    def update_input(self, human_u):
        self.human_u = human_u

    def forward(self, t, x):
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        u = self.u

        q1 = x[0, 0]
        dq1 = x[0, 1]
        h_q2 = x[0, 2]
        h_dq2 = x[0, 3]
        r_d2 = x[0, 4]
        r_dd2 = x[0, 5]
        r_d3 = x[0, 6]
        r_dd3 = x[0, 7]
        r_q4 = x[0, 8]
        r_dq4 = x[0, 9]
        r_q5 = x[0, 10]
        r_dq5 = x[0, 11]

        tau_1 = 0
        tau_3 = 0
        tau_4 = u  # robot torque
        if self.human_u is not None:
            # TODO: implement human input from input
            # Warning: change this every time you change the input
            # tau_2 = 5 * torch.sin(2 * 2 * torch.pi * (self.current_t + t) / 100)
            tau_2 = self.sample_from_data(t, self.human_u, self.freq)
        else:
            tau_2 = 0

        # fmt: off
        (
            dq1, ddq1,
            h_dq2, h_ddq2,
            r_dd2, r_ddd2,
            r_dd3, r_ddd3,
            r_dq4, r_ddq4,
            r_dq5, r_ddq5,
        ) = hri_ode_c(
            self.params,
            q1, dq1,
            h_q2, h_dq2,
            r_d2, r_dd2,
            r_d3, r_dd3,
            r_q4, r_dq4,
            r_q5, r_dq5,
            tau_1,
            tau_2,
            tau_3,
            tau_4,
        )
        # fmt: on

        output = torch.zeros_like(x)  # size (*, 2), h_ddq2, r_ddq2
        output[..., 0] = dq1
        output[..., 2] = h_dq2
        output[..., 4] = r_dd2
        output[..., 6] = r_dd3
        output[..., 8] = r_dq4
        output[..., 10] = r_dq5
        output[..., 1] = ddq1
        output[..., 3] = h_ddq2
        output[..., 5] = r_ddd2
        output[..., 7] = r_ddd3
        output[..., 9] = r_ddq4
        output[..., 11] = r_ddq5

        if squeeze:
            output = output.squeeze(0)
        return output

    def sample_from_data(self, ts, data, freq):
        # Calculate the time of each sample
        time_stamps = np.arange(0, len(data) / freq, 1 / freq)

        # Find the two nearest sampling points
        idx = self.binary_search_left(time_stamps, ts)

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

    def binary_search_left(self, time_stamps, ts):
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


if __name__ == "__main__":
    params = {
        "m1": 7.275,
        "m2": 3.75,
        "m3": 2,
        "m4": 2,
        "g": 10,
        "I_G1z": 0.121,
        "I_G2z": 0.055,
        "I_G3z": 0.02,
        "I_G4z": 0.02,
        "l1": 0.4,
        "l2": 0.4,
        "l3": 0.2,
        "l4": 0.2,
        "lc1": 0.173,
        "la1": 0.2,
        "lb1": 0.2,
        "lc2": 0.173,
        "lc3": 0.1,
        "lc4": 0.1,
        "la4": 0.2,
        "la2": 0.2,
        "K_AFz": 2000,
        "K_AFx": 4000,
        "K_AMy": 20,
        "K_BFz": 2000,
        "K_BFx": 2000,
        "K_BMy": 20,
        "D_AFz": 100,
        "D_AFx": 100,
        "D_AMy": 10,
        "D_BFz": 100,
        "D_BFx": 100,
        "D_BMy": 10,
    }

    model = HRIDx_Sim(params, torch.tensor([3.0]))

    ts = torch.linspace(0.0, 1.0, 100)
    x0 = torch.tensor(
        [torch.pi / 2, 0, -torch.pi / 2, 0, 0, 0, 0, 0, 0, 0, -torch.pi / 2, 0]
    )
    x = odeint(model, x0, ts, method="dopri5").detach().numpy()
    h_th = torch.tensor(x[:, 2])
    r_th = torch.tensor(x[:, 10])

    plt.plot(ts, h_th, label="Human")
    plt.plot(ts, r_th, label="Robot")
    plt.legend()
    plt.show()
