from os.path import join

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import sys

from prepare_variables import get_variables

YEAR = "2018"
SEASON = "jan" # jan or apr or 0
FIGURE_PATH = "figures"

def pairplots(site, data, products):
    data = np.concatenate(
        [
            data.values[:, [2, 3]],
            products[:, None],
            data.values[:, 4],
        ],
        axis=1,
    )
    sns.pairplot(
        pd.DataFrame(data, columns=["ice", "snow", "wind", "vv"])
    )
    # plt.show()


def get_posterior(vars, xs, y):
    *as_, std = vars
    n = len(as_) + 2

    for i, a in enumerate(as_):
        new_shape = np.ones(n, dtype=int)
        new_shape[i] = -1
        as_[i] = a.reshape(new_shape)

    new_shape = np.ones(n, dtype=int)
    new_shape[-2] = -1
    std = std.reshape(new_shape)

    new_shape = np.ones(n, dtype=int)
    new_shape[-1] = -1
    xs = [x.reshape(new_shape) for x in xs]
    y = y.reshape(new_shape)

    posterior = gaussian(
        x=y,
        mean=np.sum([a*x for a, x in zip(as_, xs)], axis=0),
        std=std,
    )
    posterior = np.prod(posterior, axis=-1)

    return posterior


def get_stats(posterior, vars, null_dims, print_=True, problem=None):
    # Maximum posterior probability
    argmax = np.argmax(posterior)
    prob_max = posterior.flatten()[argmax]
    # Mean and std of each parameter's marginal distribution at prob_max
    unravel_argmax = list(np.unravel_index(argmax, posterior.shape))
    vars_max = [var[unravel_argmax[i]] for i, var in enumerate(vars)]
    probs_mar = marginal_distributions(unravel_argmax, posterior)
    std_mar = [weighted_std(var, prob) for var, prob in zip(vars, probs_mar)]
    # prob_uniform = posterior.mean()

    # Maximum posterior probability for the null hypothesis (same model with null_dims)
    prob_null = get_prob_null(posterior, vars, null_dims)

    # if print_:
    #     print(" "*3, f"{problem}:")
    #     print(" "*7, "Best fit mean:", ["{:.3f}".format(x) for x in vars_max])
    #     print(" "*7, "Best fit std:", ["{:.3f}".format(x) for x in std_mar])
    #     # print(" "*7, "Rapport à l'uniforme:", prob_max / prob_uniform)
    #     print(" "*7, "Rapport à l'hypothèse nulle:", "{:.0f}".format(prob_max / prob_null), ", ({:.0E})".format(prob_max / prob_null))

    return argmax, prob_max, unravel_argmax, vars_max, probs_mar, std_mar, prob_null

"""Print stats info to console and save as text file"""
def stats_info(vars_max, std_mar, prob_max, prob_null, site, problem, log_path=0):
    info = [
        " "*3 + f"{problem}:",
        " "*7 + "Best fit mean: " + ", ".join(["{:.3f}".format(x) for x in vars_max]),
        " "*7 + "Best fit std: " + ", ".join(["{:.3f}".format(x) for x in std_mar]),
        " "*7 + "Rapport à l'hypothèse nulle: " + "{:.0f}".format(prob_max / prob_null) + " ({:.0E})".format(prob_max / prob_null)
    ]
    if log_path:
        f = open(log_path, "a")
    for line in info:
        print(line)
        if log_path:
            f.write(line+"\n")
    if log_path:
        f.close()

"""Save parameter mean and std at max posterior prob"""
def save_param(vars_max, std_mar, var_names, site, problem):
    save_name = f"param_{problem}_{site}"
    if YEAR:
        save_name += f"_{YEAR}"
    if SEASON:
        save_name += f"_{SEASON}"
    save_path = join(FIGURE_PATH, save_name+".csv")
    save_data = pd.DataFrame({"param":var_names, "mean":vars_max, "std":std_mar})
    save_data.to_csv(save_path)

"""Initiate a log text file with header, return log_path"""
def init_log(site, YEAR, SEASON, files):
    print(files)
    log_name = f"log_{site}"
    if YEAR:
        log_name += f"_{YEAR}"
    if SEASON:
        log_name += f"_{SEASON}"
    log_path = join(FIGURE_PATH, log_name+".txt")
    f = open(log_path,"w+")
    f.write("Log for bayesian inference on files \n")
    for file in files:
        f.write(file+"\n")
    f.close()

    return log_path

""" Reduces the posterior probability of a model to the special null hypothesis
case where one or more of its parameters are set to zero. Returns the max."""
def get_prob_null(posterior, vars, null_dims):
    for dim in sorted(null_dims)[::-1]:
        posterior = np.moveaxis(posterior, dim, -1)
        idx_null = np.argmin(np.abs(vars[dim]))
        posterior = posterior[..., idx_null]
    return posterior.max()

def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


def marginal_distributions(argmax, posterior):
    distributions = []
    for i, (var, name) in enumerate(zip(vars, var_names)):
        axis = tuple(j for j in range(len(posterior.shape)) if i != j)
        distributions.append(np.sum(posterior, axis=axis))
    return distributions


def gaussian(x, mean, std):
    return (
        np.exp(-((x-mean)/std)**2 / 2)
        / (np.sqrt(2*np.pi) * std)
    )


def gaussian_fill_between(a, b, std, xlim=None, ylim=None):
    if xlim is None:
        xlim = plt.xlim()
    if ylim is None:
        ylim = plt.ylim()

    # Alternative representation.
    # for s, a in zip(range(1, 4), [.25, .125, .0625]):
    #     plt.fill_between(
    #         p,
    #         line-s*snow_noise,
    #         line+s*snow_noise,
    #         alpha=a,
    #         color="tab:blue",
    #     )
    #     plt.imshow(gaussian_fill_between)

    extent = [*xlim, *ylim]
    x, y = np.meshgrid(np.linspace(*xlim, 1000), np.linspace(*ylim, 1000))
    fill_between = gaussian(y, a*x+b, std)

    tab_blue = mpl.colors.to_rgb(mpl.colors.BASE_COLORS["k"])
    alpha = np.linspace(0, .3, 1000)
    cmap = mpl.colors.ListedColormap([[*tab_blue, a] for a in alpha])
    plt.imshow(
        fill_between,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
    )

def plot_linear_dependency(problem, x, y, a, b, std, xlabel="", ylabel=""):
    fig = plt.figure(figsize=(6,4))
    plt.title(f"Site {site}")
    plt.scatter(x, y, s=4, c="k")
    extend = x.max() - x.min()
    x_line = np.linspace(x.min()-extend, x.max()+extend, 2)
    line = a*x_line + b
    gaussian_fill_between(a, b, std)
    plt.autoscale(False)
    plt.plot(x_line, line, ls='--', c="k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_name = f"Results_{problem}_{site}"
    if YEAR:
        save_name += f"_{YEAR}"
    if SEASON:
        save_name += f"_{SEASON}"
    fig.savefig(join(FIGURE_PATH, save_name),transparent=False, dpi=300)    # plt.show()
    plt.close()

def export_xy(problem, x, y, a, b, std, xlabel="", ylabel=""):
    # write data to txt file
    to_save = pd.DataFrame({"x":x, "y":y})
    save_path = f"data_analyzed/{problem}_{site}_{YEAR}.txt"
    to_save.to_csv(save_path)
    with open(save_path, 'a') as f:
        f.write(f"# {problem} x y values for Bayesian linear regression on {site} in {YEAR} \n")
        f.write(f"# x = {xlabel}, y = {ylabel} \n")

def plot_parameters(site, problem, vars, var_names, probs_mar):
    if len(var_names) == 4:
        fig = plt.figure(figsize=(11.5, 3))
    else:
        fig = plt.figure(figsize=(8.5, 3))
    mpl.rcParams["font.family"] = "Arial"
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.92, hspace=0.2, wspace=0.2) # adjust the box of axes regarding the figure size
    gs = mpl.gridspec.GridSpec(1, len(var_names))
    axes = list()
    for i in np.arange(len(var_names)):
        axes.append(plt.subplot(gs[i]))

    axes[0].set_ylabel(
        r"Marginal probability $\frac{p(\theta)}{p_{max}(\theta)}$"
    )

    for i, (var, name) in enumerate(zip(vars, var_names)):
        ax = axes[i]
        width = np.diff(var)
        probs_mar_ = probs_mar[i] / probs_mar[i].max()
        ax.bar(var, probs_mar_, [*width, width[-1]], color=[.3, .3, .3])
        ax.set_xlabel(name)
        ax.set_xlim([var.min(), var.max()])
        if i == len(vars)-1:
            ax.set_xscale('log')
            ax.set_xlim(0.1,10)
        if i > 0:
            ax.get_yaxis().set_ticklabels([])
        else:
            if site == "D":
                bay = "Deception Bay"
            elif site == "S":
                bay = "Salluit"
            elif site == "K":
                bay = "Kangiqsujuaq"
            ax.annotate(f"{bay}\n{YEAR}", xy=(0.9, 0.85), xycoords="axes fraction", ha="right", color="k")

        ax.tick_params(direction='in',which="both",right=1,top=0)
    save_name = f"Parameters_{problem}_{site}"
    if YEAR:
        save_name += f"_{YEAR}"
    if SEASON:
        save_name += f"_{SEASON}"

    fig.savefig(join(FIGURE_PATH, save_name),transparent=False, dpi=300)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    STEPS = 64  # NOTE Reduce step size to make computations faster.
    if (YEAR == "2016") & (SEASON != "apr"):
        print("Not enough data for all year 2016 or jan 2016")
        sys.exit()

    for site, data, data_files in get_variables(year=YEAR, season=SEASON):

        print(f"Site {site}")
        log_path = init_log(site, YEAR, SEASON, data_files)

        snow, ice, wind, vv = data[["snow", "ice", "wind", "VV"]].values.T

        """Snow"""

        product_dep = np.linspace(-.1, .5, STEPS)
        snow_0 = np.linspace(-1, 1, STEPS)
        snow_noise = np.logspace(-.5, 0.5, STEPS)
        vars = [product_dep, snow_0, snow_noise]
        var_names = [r"$\alpha$", r"$h_{s_{0}}$", r"$\eta_s$"]

        posterior = get_posterior(vars, [wind, np.ones_like(snow)], snow)
        argmax, prob_max, unravel_argmax, vars_max, probs_mar, std_mar, prob_null = get_stats(
            posterior, vars, null_dims=[0], problem="Snow",
        )
        stats_info(vars_max, std_mar, prob_max, prob_null,
            site=site, problem="Snow", log_path=log_path)
        save_param(vars_max, std_mar, var_names, site, problem="Snow")

        plot_linear_dependency(
            "snow",
            wind,
            snow,
            *vars_max,
            xlabel="Wind dependency",
            ylabel="Snow",
        )
        export_xy(
            "snow",
            wind,
            snow,
            *vars_max,
            xlabel="Wind dependency",
            ylabel="Snow",
        )
        plot_parameters(site, "snow", vars, var_names, probs_mar)

        """Ice"""

        snow_dep = np.linspace(-1.5, 1, STEPS)
        ice_0 = np.linspace(-1, 1, STEPS)
        ice_noise = np.logspace(-.5, 1, STEPS)
        vars = [snow_dep, ice_0, ice_noise]
        var_names = [r"$\beta$", r"$h_{i_{0}}$", r"$\eta_i$"]

        posterior = get_posterior(vars, [snow, np.ones_like(ice)], ice)
        argmax, prob_max, unravel_argmax, vars_max, probs_mar, std_mar, prob_null = get_stats(
            posterior, vars, null_dims=[0], problem="Ice",
        )
        stats_info(vars_max, std_mar, prob_max, prob_null,
            site=site, problem="Ice", log_path=log_path)
        save_param(vars_max, std_mar, var_names, site, problem="Ice")

        plot_linear_dependency(
            "ice",
            snow,
            ice,
            *vars_max,
            xlabel="Snow",
            ylabel="Ice",
        )
        plot_parameters(site, "ice", vars, var_names, probs_mar)

        """VV"""
        vv_snow_dep = np.linspace(-2, 1.5, STEPS/2)
        vv_ice_dep = np.linspace(-2, 1.5, STEPS/2)
        vv_0 = np.linspace(-1, 1, STEPS/2)
        vv_noise = np.logspace(-.5, 1, STEPS/2)
        vars = [vv_snow_dep, vv_ice_dep, vv_0, vv_noise]
        var_names = [
            r"$\gamma$",
            r"$\delta$",
            r"$\sigma_{VV~0}$",
            r"$\eta_{\sigma_{VV}}$",
        ]

        posterior = get_posterior(
            vars,
            [snow, ice, np.ones_like(vv)],
            vv,
        )
        # In this case, the null hypothesis is neither ice nor snow having an
        # effect on vv.
        argmax, prob_max, unravel_argmax, vars_max, probs_mar, std_mar, prob_null = get_stats(
            posterior, vars, null_dims=[0, 1], problem="VV",
        )
        stats_info(vars_max, std_mar, prob_max, prob_null,
            site=site, problem="VV", log_path=log_path)
        save_param(vars_max, std_mar, var_names, site, problem="VV")

        vv_snow_dep, vv_ice_dep, vv_0, vv_noise = vars_max
        plot_linear_dependency(
            "vv_to_snow",
            snow,
            vv-vv_ice_dep*ice,
            a=vv_snow_dep,
            b=vv_0,
            std=vv_noise,
            xlabel="Snow",
            ylabel=r"$\sigma_{VV}$ - Ice dependency * Ice",
        )
        plot_linear_dependency(
            "vv_to_ice",
            ice,
            vv-vv_snow_dep*snow,
            a=vv_ice_dep,
            b=vv_0,
            std=vv_noise,
            xlabel="Ice",
            ylabel=r"$\sigma_{VV}$ - Snow dependency * Snow",
        )
        plot_parameters(site, "vv", vars, var_names, probs_mar)
