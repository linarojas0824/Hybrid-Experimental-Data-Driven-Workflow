"Source: https://github.com/IBM/atgp-public/blob/master/AT-GP/models/AdaptiveTransferKernel.py"

import numpy as np
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin
from skopt.learning.gaussian_process.kernels import Kernel, Hyperparameter, Matern

class AdaptiveTransferKernel(NormalizedKernelMixin, StationaryKernelMixin, Kernel):

    def __init__(self,
                    kernel=1.0 * Matern(length_scale=1.0, nu=2.5),
                    lamb=2.0,
                    lamb_bounds=(1.0, 3.0),
                    different_noises=False,
                    source_noise_level=1.0,
                    source_noise_bounds=(1e-5, 1e-1),
                    target_noise_level=1.0,
                    target_noise_bounds=(1e-6, 1e-5)):
            self.lamb = lamb
            self.kernel = kernel
            self.noise_source_level = source_noise_level
            self.noise_target_level = target_noise_level
            self.lamb_bounds = lamb_bounds
            self.source_noise_bounds = source_noise_bounds
            self.target_noise_bounds = target_noise_bounds
            self.different_noises = different_noises

    def _f(self, X, Xp, eval_gradient=False):
            X = np.asarray(X, dtype=float)
            Xp = np.asarray(Xp, dtype=float)

            X_feat = X[:, :-1]
            Xp_feat = Xp[:, :-1]

            # domain labels
            dX = X[:, -1]
            dXp = Xp[:, -1]

            # pairwise same-domain mask
            is_same = (dX[:, None] == dXp[None, :])

            # scaling matrix: 1 if same domain, (lamb-2) if different
            matrix_lamb = is_same.astype(float) + (self.lamb - 2.0) * (~is_same).astype(float)

            noise_matrix_source = 0.0
            noise_matrix_target = 0.0
            if self.different_noises:
                # diagonal-like matrices (exact equality)
                eq = np.array([[np.array_equal(x, xp) for xp in Xp] for x in X], dtype=float)
                src = (dX[:, None] == 0.0) & (dXp[None, :] == 0.0)
                tgt = (dX[:, None] == 1.0) & (dXp[None, :] == 1.0)
                noise_matrix_source = self.noise_source_level * (eq * src)
                noise_matrix_target = self.noise_target_level * (eq * tgt)

            if eval_gradient:
                # sklearn/skopt only request gradients for K(X,X) => Xp should equal X in that call
                k_base, g = self.kernel(X_feat, eval_gradient=True)

                # gradient wrt noises (if enabled): indicator matrices
                if self.different_noises:
                    g = np.dstack((g, noise_matrix_source))
                    g = np.dstack((g, noise_matrix_target))

                # gradient wrt lamb: only affects cross-domain pairs: d/dlamb [(lamb-2) * k_base] = 1*k_base
                g_lamb = (~is_same).astype(float) * k_base
                g = np.dstack((g, g_lamb))

                k = matrix_lamb * k_base
                if self.different_noises:
                    k = k + noise_matrix_source + noise_matrix_target

                return k, g
            
            k = self.kernel(X_feat, Xp_feat)
            k = matrix_lamb * k
            if self.different_noises:
                k = k + noise_matrix_source + noise_matrix_target
            return k
        
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=float)
        if Y is None:
            return self._f(X, X, eval_gradient=eval_gradient)
        Y = np.asarray(Y, dtype=float)
        if eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        return self._f(X, Y, eval_gradient=False)

    def diag(self, X):
        X = np.asarray(X, dtype=float)
        res = self.kernel.diag(X[:, :-1])
        if self.different_noises:
            is_target = X[:, -1].astype(bool)  # FIX: last column is domain
            res = res + self.noise_target_level * is_target + self.noise_source_level * (~is_target)
        return res

    def get_params(self, deep=True):
        params = {}
        if deep:
            params.update((k, v) for k, v in self.kernel.get_params().items())
        params.update(
            source_noise_level=self.noise_source_level,
            target_noise_level=self.noise_target_level,
            lamb=self.lamb,
            different_noises=self.different_noises,
        )
        return params


    @property
    def hyperparameters(self):
        res = list(self.kernel.hyperparameters)
        if self.different_noises:
            res.append(Hyperparameter("noise_source_level", "numeric", self.source_noise_bounds))
            res.append(Hyperparameter("noise_target_level", "numeric", self.target_noise_bounds))
        res.append(Hyperparameter("lamb", "numeric", self.lamb_bounds))
        return res

    @property
    def theta(self):
        res = self.kernel.theta
        if self.different_noises:
            res = np.append(res, np.log(self.noise_source_level))
            res = np.append(res, np.log(self.noise_target_level))
        res = np.append(res, np.log(self.lamb))
        return res


    @theta.setter
    def theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        if self.different_noises:
            self.kernel.theta = theta[:-3]
            self.noise_source_level = np.exp(theta[-3])
            self.noise_target_level = np.exp(theta[-2])
            self.lamb = np.exp(theta[-1])
        else:
            self.kernel.theta = theta[:-1]
            self.lamb = np.exp(theta[-1])