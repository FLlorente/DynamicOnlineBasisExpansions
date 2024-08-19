#%%
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax, jit, value_and_grad
from functools import partial
import optax 
import numpy as np
from jax.scipy.special import logsumexp
from scipy.stats import norm

from DOEBE.doebe import DOEBE
from DOEBE.models import DOLinear_selected

import argparse

import cvxopt
from cvxopt import matrix, solvers
cvxopt.solvers.options['show_progress'] = False

parser = argparse.ArgumentParser(
    prog="subsetreg",
)
parser.add_argument("-N", "--num_data", default=6000, type=int)
parser.add_argument("-n", "--num_pretrain", default=1000, type=int)
parser.add_argument("-r", "--rand_seed", default=0, type=int)
parser.add_argument("-o", "--compute_ons", default=1, type=int)
parser.add_argument("-s", "--setting", default='open', type=str)


args = parser.parse_args()

N = args.num_data
my_seed = args.rand_seed
n_pre = args.num_pretrain
compute_ons = args.compute_ons
setting = args.setting

print(f"Running setting {setting} with seed {my_seed} and (N,n) = ({N},{n_pre})")


#%% Auxiliary functions
def get_weights_expgrad(alpha,pll_t):
    def _step_weights(carry, i):
            log_w = carry

            li = jnp.asarray(pll_t)[:,i-1]

            # Exponentiated Gradients
            log_w = log_w + alpha * jnp.exp(
                li - jax.scipy.special.logsumexp(log_w + li)
            )
            log_w = log_w - jax.scipy.special.logsumexp(log_w)

            return log_w, log_w

    logw = jnp.log(jnp.ones(pll_t.shape[0])/pll_t.shape[0])
    w = jnp.exp(logw)    
    final_log_w, log_ws = lax.scan(
            _step_weights, logw, jnp.arange(1, pll_t.shape[1] + 1)
        )

    # optimized log-weights
    log_ws = jnp.concatenate([logw.reshape(1, -1), log_ws[:-1]], axis=0)  # we don't consider the last weight

    return log_ws


def get_weights_expgrad_bma(alpha,pll_t):
    def _step_weights(carry, i):
            log_w = carry

            li = jnp.asarray(pll_t)[:,i-1]

            # Exponentiated Gradients
            log_w = log_w + alpha * li
            log_w = log_w - jax.scipy.special.logsumexp(log_w)

            return log_w, log_w

    logw = jnp.log(jnp.ones(pll_t.shape[0])/pll_t.shape[0])
    w = jnp.exp(logw)    
    final_log_w, log_ws = lax.scan(
            _step_weights, logw, jnp.arange(1, pll_t.shape[1] + 1)
        )

    # optimized log-weights
    log_ws = jnp.concatenate([logw.reshape(1, -1), log_ws[:-1]], axis=0)

    return log_ws


def get_static_weights(pll_t): # best constant rebalanced portfolio (BCRP)

    def neg_log_wealth(log_weights,pll_t):   # log_weights are not normalized!
        log_mix = logsumexp(log_weights + pll_t.T - logsumexp(log_weights),axis=1)

        return -jnp.sum(log_mix)
    
    neg_log_wealth_jit = jit(neg_log_wealth)

    init_params = jnp.log(jnp.ones(pll_t.shape[0]) \
                        + 0.3*jax.random.normal(jax.random.PRNGKey(my_seed+20),(pll_t.shape[0],))
                        )


    # Define the optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(init_params)

    # Define the update step
    @partial(jax.jit)
    def update(params, opt_state, pll_t):
        loss, grads = value_and_grad(neg_log_wealth_jit)(params,pll_t)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    # Training loop
    params = init_params
    num_steps = 1000
    loss_vals = []
    for step in range(num_steps):
        loss, params, opt_state = update(params, opt_state, pll_t)
        loss_vals.append(loss)

    static_weights = jnp.exp(params - logsumexp(params))

    return static_weights


# Modified from the Universal Portfolios library 
# https://github.com/Marigold/universal-portfolios/blob/master/universal/algos/ons.py
# Available under MIT License
class ONS:
    def __init__(self, delta=1/8, beta=1e-5, eta=0.0):
        """
        :param delta, beta, eta: Model parameters. See paper.
        """
        super().__init__()
        self.delta = delta
        self.beta = beta
        self.eta = eta

    def init_weights(self, m):
        self.A = np.mat(np.eye(m))
        self.b = np.mat(np.zeros(m)).T

        return np.ones(m) / m
        
    def step(self, r, p):
        # calculate gradient
        grad = np.mat(r / np.dot(p, r)).T
        # update A
        self.A += grad * grad.T
        # update b
        self.b += (1 + 1.0 / self.beta) * grad

        # projection of p induced by norm A
        pp = self.projection_in_norm(self.delta * self.A.I * self.b, self.A)
        return pp * (1 - self.eta) + np.ones(len(r)) / float(len(r)) * self.eta

    def projection_in_norm(self, x, M):
        """Projection of x to simplex indiced by matrix M. Uses quadratic programming."""
        m = M.shape[0]

        P = matrix(2 * M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m, 1)))
        A = matrix(np.ones((1, m)))
        b = matrix(1.0)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol["x"])
    
def get_weights_ons(delta,beta,eta, pll_t):
    reward_t = np.exp(pll_t)
    max_reward_t = reward_t.max(0, keepdims=True)
    reward_t_norm = reward_t /max_reward_t

    ons = ONS(delta, beta, eta)
    current_weights = ons.init_weights(pll_t.shape[0])
    all_weights = [current_weights]

    for current_reward in reward_t_norm.T:
        current_weights = ons.step(current_reward, current_weights)
        all_weights.append(current_weights)

    all_weights = np.maximum(np.stack(all_weights), np.zeros_like(np.stack(all_weights))+1e-64)

    return all_weights    


#%% Generating the data
J = 15
h = 5
gamma = 0.0285

X_tr = 5 + 1.0*jax.random.normal(jax.random.PRNGKey(my_seed),(N,J))

beta_gt = gamma*( 
    (np.abs(np.arange(1,J+1) - 4)<h)*(h - np.abs(np.arange(1,J+1) - 4))**2 + \
    (np.abs(np.arange(1,J+1) - 8)<h)*(h - np.abs(np.arange(1,J+1) - 8))**2 + \
    (np.abs(np.arange(1,J+1) - 12)<h)*(h - np.abs(np.arange(1,J+1) - 12))**2 
) 

Y_tr = X_tr @ beta_gt + 1.0*jax.random.normal(jax.random.PRNGKey(my_seed+10),(N,))


#%% 
'''
M-open scenario:    J one-dimensional models
M-closed scenario:  J models using the first k covariates, k=1,...,J
'''

if setting == "open":
    # one dimensional linear regression models
    doelinear = DOEBE(
         [DOLinear_selected(1, 1.0, 0, 0.25, active_dimensions= np.array([k]),bias=False) for k in range(J)],
         min_weight=0.0
         )  
    doelinear.pretrain(X_tr[:n_pre],Y_tr[:n_pre])  
    _, _, ws, yhat, cov_yhat = doelinear.fit(X_tr[n_pre:],Y_tr[n_pre:],return_ws=True)

elif setting == "closed":
    doelinear = DOEBE(
         [DOLinear_selected(k+1, 1.0, 0, 0.25, active_dimensions= np.arange(k),bias=False) for k in range(J)],
         min_weight=0.0
         )  
    doelinear.pretrain(X_tr[:n_pre],Y_tr[:n_pre])  
    _, _, ws, yhat, cov_yhat = doelinear.fit(X_tr[n_pre:],Y_tr[n_pre:],return_ws=True)

else:
     raise Exception("-s must be 'open' or 'closed'.")    


# computing the pll_t
pll_t = np.empty((J,N - n_pre))
for j in range(J):
    pll_t[j,:] = norm.logpdf(Y_tr[n_pre:],yhat[:,j],np.sqrt(cov_yhat[:,j]))


rewards = {}
weights = {}

# computing the weights
logws_eg = get_weights_expgrad(1e-2, pll_t)
weights["eg"] = logws_eg

logws_bma = get_weights_expgrad_bma(1, pll_t)
weights["bma"] = logws_bma

static_weights = get_static_weights(pll_t)
weights["static"] = static_weights

# computing rewards
reward_t_eg = np.cumsum(logsumexp(pll_t.T + logws_eg, axis=1))/ \
    np.arange(1,N - n_pre + 1)
rewards["eg"] = reward_t_eg

reward_t_bma=  np.cumsum(logsumexp(pll_t.T + logws_bma, axis=1))/ \
    np.arange(1,N - n_pre + 1)
rewards["bma"] = reward_t_bma

reward_t_static = np.cumsum(logsumexp(pll_t.T + np.log(static_weights), axis=1))/  \
    np.arange(1, N - n_pre + 1)
rewards["static"] = reward_t_static

if compute_ons:
     # normalizing the rewards (exponentiated log predictive values)
     ws_ons = get_weights_ons(0.8,1e-2,0.01, pll_t)
     weights["ons"] = ws_ons

     reward_t_ons =  np.cumsum(logsumexp(pll_t.T + np.log(ws_ons[:-1]), axis=1))/ \
        np.arange(1,N - n_pre + 1)
     rewards["ons"] = reward_t_ons

     np.savez(f'{setting}/results_setting_{setting}_seed_{my_seed}.npz', logws_eg, reward_t_eg, logws_bma, reward_t_bma, static_weights, reward_t_static, ws_ons, reward_t_ons)
else:
     np.savez(f'{setting}/results_setting_{setting}_seed_{my_seed}_no_ons.npz', logws_eg, reward_t_eg, logws_bma, reward_t_bma, static_weights, reward_t_static)



