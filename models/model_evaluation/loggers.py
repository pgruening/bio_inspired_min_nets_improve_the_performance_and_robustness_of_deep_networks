import numpy as np
import torch
from DLBio.pt_train_printer import IPrinterFcn


class MeanActivationLogger(IPrinterFcn):
    def __init__(self, act_getter):
        self.act_getter = act_getter
        self.values = []

    def update(self, *args):
        with torch.no_grad():
            self.values.append(
                self.act_getter.out.mean([0, 2, 3]).cpu().numpy()
            )
            self.act_getter.out = None

        return self

    def restart(self):
        self.values = []

    def __call__(self):
        X = np.stack(self.values, -1).copy()
        X = list(np.mean(X, -1))
        return [float(x) for x in X]


class MeanAbsActivationLogger(MeanActivationLogger):

    def update(self, *args):
        with torch.no_grad():
            if self.act_getter.out.ndim == 4:
                self.values.append(
                    torch.abs(self.act_getter.out).mean(
                        [0, 2, 3]).cpu().numpy()
                )
            elif self.act_getter.out.ndim == 2:
                self.values.append(
                    torch.abs(self.act_getter.out).mean(0).cpu().numpy()
                )
            else:
                raise ValueError(f'Unknown dim: {self.act_getter.out.ndim}')
            self.act_getter.out = None

        return self
