from experiment_utils import *
import jax.numpy as jnp
import jax
from DOEBE.doebe import DOEBE
from DOEBE.dosbe import DOSBE
from DOEBE.models import *
from tqdm import tqdm
import numpy as np
from scipy.cluster.vq import kmeans2
import seaborn as sns

jax.config.update("jax_enable_x64", True)
sns.set_style("whitegrid")
sns.set_palette("colorblind")


def make_models(X, n_features=100):
    d = X.shape[1]
    M = 100 // d

    L = np.max([np.abs(np.min(X, axis=0)), np.abs(np.max(X, axis=0))], axis=0) * 1.5

    ls_guesses = [
        (jnp.max(X, axis=0) - jnp.min(X, axis=0)) / f for f in [0.1, 1.0, 10.0]
    ]

    doebe_gp = DOEBE(
        [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ],
        min_weight=0.0,
    )

    dosbe_gp_m3 = DOSBE(
        [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ],
        alpha=1e-3,
    )
    dosbe_gp_m2 = DOSBE(
        [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ],
        alpha=1e-2,
    )

    dosbe_gp_m1 = DOSBE(
        [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ],
        alpha=1e-1,
    )

    dosbe_gp_m0 = DOSBE(
        [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ],
        alpha=1.0,
    )

    dosbe_gp_ep = DOSBE(
        [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 1e-3, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ]
        + [
            DOAddHSGP(
                L, M, d, jnp.ones(d) * ls_guess, 1.0, 0.0, 0.25, kernel_type="rbf"
            )
            for ls_guess in ls_guesses
        ],
        alpha=1e-2,
        strategy="eg_fixedshare",
    )

    return [
        doebe_gp,
        dosbe_gp_m3,
        dosbe_gp_m2,
        dosbe_gp_m1,
        dosbe_gp_m0,
        dosbe_gp_ep,
    ], [
        "DOEBE-GP",
        r"DOSBE-GP ($10^{-3})",
        r"DOSBE-GP ($10^{-2})",
        r"DOSBE-GP ($10^{-1})",
        r"DOSBE-GP ($10^{0})",
        r"DOSBE-GP (Fixed Share)",
    ]


def pretrain(models, X, y, lrs, sample_type, n_samples=1000):
    for model_idx, model in enumerate(tqdm(models)):
        model.pretrain_and_sample(
            X[:n_samples],
            y[:n_samples],
            lrs[model_idx],
            sampling_type=sample_type[model_idx],
            verbose=False,
            n_samples=1,
        )


def fit(models, X, y, n_pretrain=1000, n_batch=2500):
    yhats = []
    yvars = []
    mls = []
    for model in tqdm(models):
        yhat, yvar, ml = model.fit_minibatched(
            X[n_pretrain:], y[n_pretrain:], n_batch=n_batch
        )
        yhats.append(yhat)
        yvars.append(yvar)
        mls.append(ml)

    return yhats, yvars, mls


if __name__ == "__main__":
    datasets = [
        "Friedman #1",
        "Friedman #2",
        "Elevators",
        "SARCOS",
        "Kuka #1",
        "CaData",
        "CPU Small",
    ]
    N_to_pretrain = 1000
    N_trials = 1
    N_models = 6

    yhat_collection = []
    yvar_collection = []
    y_collection = []
    mixture_ls_collection = []

    model_names = ""
    for dataset in datasets:
        print(f"Loading Data for {dataset}")
        X, y = get_data(dataset)
        X = X
        y = y

        yhat_trials = np.zeros((N_trials, N_models, y.shape[0] - N_to_pretrain))
        yvar_trials = np.zeros((N_trials, N_models, y.shape[0] - N_to_pretrain))
        mixture_ls_trials = np.zeros((N_trials, N_models, y.shape[0] - N_to_pretrain))

        for trial in range(N_trials):
            print("Initializing Models")
            models, model_names = make_models(X)

            print("Pretraining Models")
            lrs = [1e-2] * 10
            sample_type = [
                "laplace",
                "laplace",
                "laplace",
                "laplace",
                "laplace",
                "laplace",
            ]
            pretrain(models, X, y, lrs, sample_type, n_samples=N_to_pretrain)

            print("Fitting Models")
            yhats, yvars, mls = fit(models, X, y, n_pretrain=N_to_pretrain)
            yhats = np.stack(yhats)
            yvars = np.stack(yvars)
            mls = np.stack(mls)
            yhat_trials[trial] = yhats
            yvar_trials[trial] = yvars
            mixture_ls_trials[trial] = mls

        pll = mixture_ls_trials.mean(axis=2)

        yhat_collection.append(yhat_trials)
        yvar_collection.append(yvar_trials)
        y_collection.append(y[N_to_pretrain:])
        mixture_ls_collection.append(mixture_ls_trials)
        plot_results_avg(
            yhat_trials,
            yvar_trials,
            y[N_to_pretrain:],
            model_names,
            dataset,
            prefix="DOSBE_Exp2",
        )

    plot_summary_nmse(
        yhat_collection, y_collection, model_names, datasets, prefix="DOSBE_Exp2"
    )
    plot_summary_pll(
        yhat_collection,
        yvar_collection,
        y_collection,
        model_names,
        datasets,
        prefix="DOSBE_Exp2",
        mixture_ls_collection=mixture_ls_collection,
    )
