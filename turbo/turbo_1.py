###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import logging
import math
# from copy import deepcopy
from typing import Optional

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

logger = logging.getLogger("turbo")


class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
            self,
            f,
            lb: np.ndarray,
            ub: np.ndarray,
            n_init: int,
            use_ard: int = True,
            max_cholesky_size: int =2000,
            n_training_steps: int =50,
            min_cuda: int =1024,
            device="cpu",
            dtype="float64",
            boundary=None
    ):
        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert n_init > 0 and isinstance(n_init, int)
        assert max_cholesky_size >= 0
        assert n_training_steps >= 30
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"

        # Save function information
        self.boundary = boundary if boundary else []
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        else:
            self.device = torch.device("cpu")

    def _create_candidates(self, n_cand, batch_size, X, fX, length, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (fX - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=self.n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub] in [0, 1]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([batch_size])).t().cpu().detach().numpy()
        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, batch_size, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((batch_size, self.dim))
        for i in range(batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = X_cand[indbest, :].copy()
            y_cand[indbest, :] = np.inf
        return X_next

    def get_samples_in_region(self, cands):
        if len(self.boundary) == 0:
            # no boundary, return all candidates
            return 1.0, cands
        elif len(cands) == 0:
            return 0.0, cands
        else:
            # with boundaries, return filtered cands
            total = len(cands)
            for node in self.boundary:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    return 0, np.array([])
                assert len(cands) > 0
                cands = cands[boundary.predict(cands) == node[1]]
                # node[1] store the direction to go
            ratio = len(cands) / total
            assert len(cands) <= total
            return ratio, cands

    def get_init_samples(self):
        num_samples = 5000
        while True:
            X_init = latin_hypercube(num_samples, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            ratio, X_init = self.get_samples_in_region(X_init)
            print("sampling for init:", X_init.shape, " target=", self.n_init)

            # print("init ratio:", ratio, num_samples)
            if len(X_init) > self.n_init:
                X_init_idx = np.random.choice(len(X_init), self.n_init)
                return X_init[X_init_idx]
            else:
                num_samples *= 2

    def optimize(self, num_samples: int, batch_size: int,
                 x_init: Optional[np.ndarray] = None, fx_init: Optional[np.ndarray] = None):
        """Run the full optimization process."""
        n_cand = min(100 * self.dim, 5000)
        # failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        # succtol = 3

        if x_init is None:
            x_init = self.get_init_samples()
        if fx_init is None:
            fx_init = self.f(x_init)
        # x = np.empty((0, self.dim))
        # fx = np.empty((0,))
        total_samples = num_samples + len(x_init)

        # Update budget and set as initial data for this TR
        # curr_x = x_init.copy()
        # curr_fx = fx_init.copy()
        # curr_fx_min = curr_fx.min()
        x = x_init.copy()
        fx = fx_init.copy()
        curr_fx_min = fx.min()
        failcount = 0
        succcount = 0
        length = self.length_init

        logger.debug(f"Starting from fbest = {curr_fx_min:.4}")

        # Thompson sample to get next suggestions
        while len(x) < total_samples and length >= self.length_min:
            # Warp inputs
            norm_x = to_unit_cube(x, self.lb, self.ub)  # project X to [lb, ub] as X was in [0, 1]

            # Standardize values
            norm_fx = fx.flatten()

            # Create th next batch
            cand_x, cand_y, _ = self._create_candidates(
                n_cand, batch_size, norm_x, norm_fx, length=length, hypers={}
            )
            next_x = self._select_candidates(batch_size, cand_x, cand_y)

            # Undo the warping
            next_x = from_unit_cube(next_x, self.lb, self.ub)

            # Evaluate batch
            next_fx = self.f(next_x)

            # Update trust region
            curr_fx_min = fx.min()
            next_fx_min = next_fx.min()
            if next_fx_min < curr_fx_min - 1e-3 * math.fabs(curr_fx_min):
                length = min([1.2 * length, self.length_max])
                # succcount += 1
                # failcount = 0
            else:
                length /= 1.2
                # succcount = 0
                # failcount += 1

            # if succcount == succtol:  # Expand trust region
            #     length = min([2.0 * length, self.length_max])
            #     succcount = 0
            # elif failcount == failtol:  # Shrink trust region
            #     length /= 2.0
            #     failcount = 0

            # Update budget and append data
            # curr_x = np.vstack((curr_x, next_x))
            # curr_fx = np.hstack((curr_fx, next_fx))

            x = np.vstack((x, next_x))
            fx = np.hstack((fx, next_fx))

            if next_fx_min < curr_fx_min:
                curr_min = next_fx_min
                logger.debug(f"{len(x)}) New best: {curr_min:.4}")

        fx = fx.ravel()
        sf = fx.argsort()
        return x[sf[:num_samples]], fx[sf[:num_samples]]
