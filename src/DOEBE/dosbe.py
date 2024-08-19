from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp
from typing import List
from .models import DOBE
import numpy as np
from jaxtyping import Float, Array
import objax
import copy
import math
from jaxopt import OSQP

from typing import Tuple


class DOSBE(objax.Module):

    def __init__(
        self, models: List[DOBE], strategy: str = "eg", alpha=1e-2, delta=1e-6
    ):
        """Creates a (dynamic) online ensemble of basis expansions

        Args:
            models (List[DOBE]): models to ensemble
            strategy (str): portfolio selection strategy. Right now, accepts `EG` for exponentiated gradients.
        """
        self.models = objax.ModuleList(models)
        self.w = objax.StateVar(jnp.ones(len(models)) / len(models))

        available_strategies = ["eg", "eg_fixedshare"]
        if strategy.lower() not in available_strategies:
            raise Exception(
                f"Strategy {strategy} not yet implemented! Currently available methods are {available_strategies.join(', ')}."
            )
        self.strategy = strategy
        self.alpha = alpha
        self.delta = delta

    def pretrain(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N 1"],
        lr: float = 1e-2,
        iters: int = 500,
        verbose: bool = False,
        jit: bool = True,
    ):
        """Pretrains models by maximizing the marginal likelihood, using Adam.

        This class will ensemble arbitrary DOBE models together. If all models are of the same type, it may be faster to refactor and vmap the pretraining.

        Args:
            X (Float[Array, &quot;N D&quot;]): input data
            y (Float[Array, &quot;N 1&quot;]): output data
            lr (float, optional): learning rate for Adam. Defaults to 1e-2.
            iters (int, optional): number of iterations for Adam. Defaults to 500.
            verbose (bool, optional): Defaults to False.
        """
        for model_idx in range(len(self.models)):
            opt = objax.optimizer.Adam(self.models[model_idx].vars())
            gv = objax.GradValues(
                self.models[model_idx].mnll, self.models[model_idx].vars()
            )
            last_val = jnp.inf

            @objax.Function.with_vars(
                self.models[model_idx].vars() + gv.vars() + opt.vars()
            )
            def train_op():
                df, f = gv(X, y)
                opt(lr, df)
                return f

            if jit:
                train_op = objax.Jit(train_op)

            for iter_idx in range(iters):
                f_value = train_op()
                if (iter_idx % 100 == 0 or iter_idx == iters - 1) and verbose:
                    print(iter_idx, f_value)

                if last_val - f_value[0] < 1e-10:
                    break
                last_val = f_value[0]

            self.models[model_idx].sigma_theta = self.models[
                model_idx
            ].var_theta * jnp.eye(self.models[model_idx].n_features)

    def pretrain_and_sample(
        self,
        X: Float[Array, "N D"],
        y: Float[Array, "N"],
        lr: float = 1e-2,
        iters: int = 500,
        verbose: bool = False,
        n_samples: int = 10,
        sampling_type: str = "laplace",
        jit=True,
    ):
        """Pretrains models by maximizing the marginal likelihood, using Adam.

        This class will ensemble arbitrary DOBE models together. If all models are of the same type, it may be faster to refactor and vmap the pretraining.

        Args:
            X (Float[Array, &quot;N D&quot;]): input data
            y (Float[Array, &quot;N 1&quot;]): output data
            lr (float, optional): learning rate for Adam. Defaults to 1e-2.
            iters (int, optional): number of iterations for Adam. Defaults to 500.
            verbose (bool, optional): Defaults to False.
            n_samples (int, optional): number of samples to generate for each model.
            sampling_type (str, optional): sample from Laplace approximation of the MLL (`"laplace"`) or a random pertubation of the trained hyperparameters (`"gaussian"`).
        """

        # First, pretrain the models
        self.pretrain(X, y, lr=lr, iters=iters, verbose=verbose, jit=jit)

        if type(sampling_type) is str:
            sampling_type = [sampling_type] * len(self.models)

        # repeat over all models in the ensemble
        for model_idx in range(len(self.models)):
            if sampling_type[model_idx] == "laplace":
                # calculate the negative Hessian of the MLL for Laplace approx (same as Hessian of NMLL)
                current_model = self.models[model_idx]
                hess = objax.Hessian(current_model.mnll, current_model.vars())(X, y)

                # Make dimensions work out nicely
                for i in range(len(hess)):
                    current_dim = jnp.atleast_2d(hess[i][i]).shape[0]
                    for j in range(len(hess)):
                        if current_dim == 1:
                            hess[i][j] = jnp.atleast_2d(hess[i][j])
                        else:
                            hess[i][j] = jnp.atleast_2d(hess[i][j]).T

                # Construct Hessians, and factor them with the SVD to generate random samples
                # Notice that since if H = U S V^*, H^-1 = V S^-1 U^*, we can speed up calculations
                # by only computing the SVD once.
                hess_matrix = jnp.block(hess)
                _, s, vh = jnp.linalg.svd(hess_matrix)
                inv_hess_factor = vh.T @ jnp.diag(
                    1 / jnp.clip(jnp.sqrt(s), a_min=1e-16)
                )

                # Make samples by making a deep copy, randomly sampling from the Laplace approx, and changing the copy
                for n_sample in range(n_samples - 1):
                    model_copy = copy.deepcopy(current_model)
                    random_sample = jnp.squeeze(
                        inv_hess_factor
                        @ objax.random.normal((inv_hess_factor.shape[0], 1))
                    )
                    d_offset = 0

                    for v_idx, v in enumerate(model_copy.vars().subset(objax.TrainVar)):
                        # TrainRefs like this are necessary for Objax internal reasons
                        ref = objax.TrainRef(v)
                        dim_v = jnp.atleast_1d(v.value).shape[0]
                        ref.value = v.value + jnp.squeeze(
                            random_sample[d_offset : d_offset + dim_v]
                        )
                        d_offset += dim_v

                    self.models.append(model_copy)
            elif sampling_type[model_idx] == "gaussian":
                # Comparitively a simpler way, just take a pertubation with WGN.
                current_model = self.models[model_idx]
                for n_sample in range(n_samples - 1):
                    model_copy = copy.deepcopy(current_model)
                    for v_idx, v in enumerate(model_copy.vars().subset(objax.TrainVar)):
                        ref = objax.TrainRef(v)
                        ref.value = v.value + objax.random.normal(v.value.shape) * 1e-3
                    self.models.append(model_copy)
            else:
                raise ValueError(
                    'Sampling type not recognized. It must be either "laplace" or "gaussian".'
                )

        # We added new models, so must redefine the weight vector
        self.w = objax.StateVar(jnp.ones(len(self.models)) / len(self.models))

        # We never updated the initial sigma_theta for each model
        for model_idx in range(len(self.models)):
            self.models[model_idx].sigma_theta = self.models[
                model_idx
            ].var_theta * jnp.eye(self.models[model_idx].n_features)

    def fit(
        self, X: Float[Array, "N D"], y: Float[Array, "N 1"], return_ws=False, **kwargs
    ) -> Tuple[Float[Array, "N 1"], Float[Array, "N 1"]]:
        """Fits models according to (online) data `X` and `y`.

        A trick used in Lu et al. is to only fit the models with nonzero
        weights. Since Jax does not work well with ragged arrays and the
        methods of this class are general, this trick is not done here. It
        may be faster to "batch" the data, where the trick is implemented.
        For example, if `w[0] == 0` after training on `X[:100]` and `y[:100]`,
        then the call `fit(X[100:], y[100:])` will only fit the models with
        nonzero weights.

        Like in `pretrain`, it would be faster to vmap/pmap over the models
        if they are of the same class.

        Args:
            X (Float[Array, &quot;N D&quot;]): training X data
            y (Float[Array, &quot;N 1&quot;]): training y data

        Returns:
            Tuple[Float[Array, &quot;N 1&quot;], Float[Array, &quot;N 1&quot;]]: The mean and variance of the predictive distribution p(y_t | x_{1:t}, y_{1:t-1})
        """
        yhats = []
        cov_yhats = []
        ls = []

        # The strategy here is to separate calculation of weights and predictive densities
        # This allows for simple implementation with lax scans
        for j, model in enumerate(self.models):
            yhat, cov_yhat, l = model.predict_and_update(X, y)

            yhats.append(yhat)
            cov_yhats.append(cov_yhat)
            ls.append(l)

        yhat = jnp.vstack(yhats).T
        cov_yhat = jnp.vstack(cov_yhats).T
        ls = jnp.vstack(ls).T

        # Next, use all predictive values to weight
        if self.strategy.lower() == "eg":
            if "alpha" in kwargs:
                alpha = kwargs["alpha"]
            elif hasattr(self, "alpha"):
                alpha = self.alpha
            else:
                alpha = self.alpha

            def _step_weights(carry, i):
                log_w = carry

                li = -ls[i - 1]  # log-likelihood l_t

                # Exponentiated Gradients
                # w_{t+1} = w_t exp(alpha l_t / sum(w l_t))
                log_w = log_w + alpha * jnp.exp(
                    li - jax.scipy.special.logsumexp(log_w + li)
                )
                # Project back to simplex with L_1 norm
                log_w = log_w - jax.scipy.special.logsumexp(log_w)

                return log_w, log_w

        elif self.strategy.lower() == "eg_fixedshare":
            if "alpha" in kwargs:
                alpha = kwargs["alpha"]
            elif hasattr(self, "alpha"):
                alpha = self.alpha
            else:
                alpha = self.alpha

            if "delta" in kwargs:
                delta = kwargs["delta"]
            elif hasattr(self, "delta"):
                delta = self.delta
            else:
                delta = 1e-6

            def _step_weights(carry, i):
                log_w = carry

                li = -ls[i - 1]  # log-likelihood l_t

                # Exponentiated Gradients
                # w_{t+1} = w_t exp(alpha l_t / sum(w l_t))
                log_w = log_w + alpha * jnp.exp(
                    li - jax.scipy.special.logsumexp(log_w + li)
                )
                # Project back to simplex with L_1 norm
                log_w = log_w - jax.scipy.special.logsumexp(log_w)

                # Apply fixed share
                log_w = jax.scipy.special.logsumexp(
                    jnp.stack(
                        [
                            jnp.log(1 - delta) + log_w,
                            jnp.log(delta)
                            - jnp.ones(log_w.shape[0]) * jnp.log(log_w.shape[0]),
                        ],
                        axis=0,
                    ),
                    axis=0,
                )

                return log_w, log_w

        else:
            raise Exception(f"Strategy {self.strategy} not implemented!")

        final_log_w, log_ws = lax.scan(
            _step_weights, jnp.log(self.w), jnp.arange(1, X.shape[0] + 1)
        )

        log_ws = jnp.concatenate([self.w.reshape(1, -1), log_ws[:-1]], axis=0)
        self.w = jnp.exp(final_log_w)

        ymean = jnp.sum(jnp.exp(log_ws) * yhat, axis=1)
        yvar = jnp.sum(
            (cov_yhat + (jnp.reshape(ymean, (-1, 1)) - yhat) ** 2) * jnp.exp(log_ws),
            axis=1,
        )
        mixture_ls = jax.scipy.special.logsumexp(log_ws + -ls, axis=1)

        if return_ws:
            return ymean, yvar, jnp.exp(log_ws)
        else:
            return ymean, yvar, mixture_ls

    def fit_minibatched(self, X, y, n_batch=2000):
        ymeans = []
        yvars = []
        mixture_ls = []

        N = X.shape[0]

        for n in range(int(math.ceil(N / n_batch))):
            ymean, yvar, ml = self.fit(
                X[n * n_batch : (n + 1) * n_batch], y[n * n_batch : (n + 1) * n_batch]
            )
            ymeans.append(ymean)
            yvars.append(yvar)
            mixture_ls.append(ml)

        ymean = jnp.concatenate(ymeans)
        yvar = jnp.concatenate(yvars)
        mls = jnp.concatenate(mixture_ls)

        return ymean, yvar, mls
