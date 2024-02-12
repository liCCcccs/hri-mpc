import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from mpc.c_funcs.hri_ode_c import func as hri_ode_c  # Import the Cython function

import numpy as np

from mpc import util

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("bmh")


class HRIDx(nn.Module):
    def __init__(self, model_params=None, mpc_param=None, simple=True):
        super().__init__()
        self.simple = simple

        self.max_torque = 24
        self.dt = 0.01
        self.n_state = 12
        self.n_ctrl = 1

        # TODO: get params from inputs

        self.goal_state = torch.Tensor([1.0, 0.0, 0.0])
        self.goal_weights = torch.Tensor([1.0, 1.0, 0.1])
        self.ctrl_penalty = 0.001
        self.lower, self.upper = -24.0, 24.0

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

        self.params = model_params

    def forward(self, x, u):
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
        h_q2 = x[0, 2]
        r_d2 = x[0, 4]
        r_d3 = x[0, 6]
        r_q4 = x[0, 8]
        r_q5 = x[0, 10]
        dq1 = x[0, 1]
        h_dq2 = x[0, 3]
        r_dd2 = x[0, 5]
        r_dd3 = x[0, 7]
        r_dq4 = x[0, 9]
        r_dq5 = x[0, 11]

        tau_1 = 0
        tau_2 = 0
        tau_3 = 0
        tau_4 = u[0]  # robot torque

        (
            dq1,
            h_dq2,
            r_dd2,
            r_dd3,
            r_dq4,
            r_dq5,
            ddq1,
            h_ddq2,
            r_ddd2,
            r_ddd3,
            r_ddq4,
            r_ddq5,
        ) = hri_ode_c(
            self.params,
            q1,
            h_q2,
            r_d2,
            r_d3,
            r_q4,
            r_q5,
            dq1,
            h_dq2,
            r_dd2,
            r_dd3,
            r_dq4,
            r_dq5,
            tau_1,
            tau_2,
            tau_3,
            tau_4,
        )

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
        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.n_ctrl)))
        assert not hasattr(self, "mpc_lin")
        px = -torch.sqrt(self.goal_weights) * self.goal_state  # + self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


if __name__ == "__main__":
    dx = HRIDx()
    n_batch, T = 1, 10
    u = torch.zeros(T, n_batch, dx.n_ctrl)
    xinit = torch.zeros(n_batch, dx.n_state)
    xinit[:, 0] = np.cos(0)
    xinit[:, 1] = np.sin(0)
    x = xinit
    for t in range(T):
        x = dx(x, u[t])
        fig, ax = dx.get_frame(x[0])
        fig.savefig("{:03d}.png".format(t))
        plt.close(fig)

    vid_file = "pendulum_vid.mp4"
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = (
        "(/usr/bin/ffmpeg -loglevel quiet "
        "-r 32 -f image2 -i %03d.png -vcodec "
        "libx264 -crf 25 -pix_fmt yuv420p {}/) &"
    ).format(vid_file)
    os.system(cmd)
    for t in range(T):
        os.remove("{:03d}.png".format(t))


class HRIDx(nn.Module):
    def __init__(self, model_params=None, mpc_param=None, simple=True):
        super().__init__()
        self.simple = simple

        self.max_torque = 24
        self.dt = 0.01
        self.n_state = 12
        self.n_ctrl = 1

        # TODO: get params from inputs

        self.goal_state = torch.Tensor([1.0, 0.0, 0.0])
        self.goal_weights = torch.Tensor([1.0, 1.0, 0.1])
        self.ctrl_penalty = 0.001
        self.lower, self.upper = -24.0, 24.0

        self.mpc_eps = 1e-3
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

        self.params = model_params

    def forward(self, x, u):
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
        h_q2 = x[0, 2]
        r_d2 = x[0, 4]
        r_d3 = x[0, 6]
        r_q4 = x[0, 8]
        r_q5 = x[0, 10]
        dq1 = x[0, 1]
        h_dq2 = x[0, 3]
        r_dd2 = x[0, 5]
        r_dd3 = x[0, 7]
        r_dq4 = x[0, 9]
        r_dq5 = x[0, 11]

        tau_1 = 0
        tau_2 = 0
        tau_3 = 0
        tau_4 = u[0]  # robot torque

        (
            dq1,
            h_dq2,
            r_dd2,
            r_dd3,
            r_dq4,
            r_dq5,
            ddq1,
            h_ddq2,
            r_ddd2,
            r_ddd3,
            r_ddq4,
            r_ddq5,
        ) = hri_ode_c(
            self.params,
            q1,
            h_q2,
            r_d2,
            r_d3,
            r_q4,
            r_q5,
            dq1,
            h_dq2,
            r_dd2,
            r_dd3,
            r_dq4,
            r_dq5,
            tau_1,
            tau_2,
            tau_3,
            tau_4,
        )

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
        q = torch.cat((self.goal_weights, self.ctrl_penalty * torch.ones(self.n_ctrl)))
        assert not hasattr(self, "mpc_lin")
        px = -torch.sqrt(self.goal_weights) * self.goal_state  # + self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


if __name__ == "__main__":
    dx = HRIDx()
    n_batch, T = 1, 10
    u = torch.zeros(T, n_batch, dx.n_ctrl)
    xinit = torch.zeros(n_batch, dx.n_state)
    xinit[:, 0] = np.cos(0)
    xinit[:, 1] = np.sin(0)
    x = xinit
    for t in range(T):
        x = dx(x, u[t])
        fig, ax = dx.get_frame(x[0])
        fig.savefig("{:03d}.png".format(t))
        plt.close(fig)

    vid_file = "pendulum_vid.mp4"
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = (
        "(/usr/bin/ffmpeg -loglevel quiet "
        "-r 32 -f image2 -i %03d.png -vcodec "
        "libx264 -crf 25 -pix_fmt yuv420p {}/) &"
    ).format(vid_file)
    os.system(cmd)
    for t in range(T):
        os.remove("{:03d}.png".format(t))
