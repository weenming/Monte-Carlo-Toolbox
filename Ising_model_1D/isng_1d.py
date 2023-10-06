import numpy as np
import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm
import dill


class Ising1D:
    def __init__(self, n, T, B=0, J=1, mu=1):

        self.n = n
        self.reset_s()

        self.J = 1
        self.mu = 1

        self.T = T
        self.B = B
        self.H = None
        self.updated = False

    def reset_s(self):
        s = t.rand(self.n)
        s[s > 0.5] = 1
        s[s <= 0.5] = -1
        self.set_s(s)

    def set_s(self, s):
        self.s = s

    def H_flip_faster(self, last_flip):
        s, i = self.s, last_flip
        H = t.t_copy(self.H)
        useful_nn = s[i - 1] * s[(i + 1) % self.n] > 0
        H -= 4 * self.J * self.s[i] * s[i - 1] * int(useful_nn)
        H -= 2 * self.mu * self.B * self.s[i]
        return H

    def get_H(self, last_flip=None):
        if last_flip is None:
            inter = (self.s[1:] * self.s[:-1]).sum() + self.s[0] * self.s[-1]
            inter = -self.J * inter
            ext = -self.mu * self.B * self.s.sum()

            H = (inter + ext)

        else:
            H = self.H_flip_faster(last_flip)
            # i = last_flip
            # s = self.s
            # except_new = self.H - (
            #     -self.J * (s[i - 1] + s[(i + 1) % self.n]) * -s[i]
            #     - self.mu * self.B * -s[i]
            # )
            # self.H = except_new + (
            #     -self.J * (s[i - 1] + s[(i + 1) % self.n]) * s[i]
            #     - self.mu * self.B * s[i]
            # )
        return H

    def init_H(self):
        self.H = self.get_H()

    def sample(self):
        i = np.random.randint(0, self.n)
        # candidate
        H = self.H
        self.s[i] *= -1
        H_new = self.get_H(i)
        # elect
        p = np.exp(-1 / self.T * (H_new - H))
        accept = np.random.random() <= p
        # flip / unflip
        if not accept:
            self.s[i] *= -1
            H_new = self.get_H(i)

        self.H = H_new

        return accept

    def run(self, iter, T=None, B=None):
        self.T = T if T is not None else self.T
        self.B = B if B is not None else self.B
        self.init_H()

        for i in tqdm(range(iter)):
            self.sample()

        dill.dump(self, open('./run', 'wb'))
