# Bayesian linear regression and model inference
# Written by Jérome Simon (jerome.simon@geolearn.ca)
# and
# Sophie Dufour-Beauséjour (s.dufour.beausejour@gmail.com)
#
# June 2020

from os.path import join

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import sys

from prepare_variables import get_variables

# YEAR = "2016"
SEASON = "apr-may" # jan-feb or apr-may or 0
DATA_PATH = "data_orbit13_win_9"
annotations = "/annotations/576m2" # "/annotations" or "" empty
BAND_NAME = "HH"
if "orbit21" in DATA_PATH:
    FIGURE_PATH = f"figures{annotations}/orbit21/{SEASON}"
elif "orbit13" in DATA_PATH:
    FIGURE_PATH = f"figures{annotations}/orbit13/{SEASON}"
else:
    FIGURE_PATH = f"figures{annotations}/C-band/{SEASON}"

"""
vars = parameter values, list of 1d arrays covering the hypothesis space
xs = data x_i plus an array of ones
y = data y
"""
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

"""From posterior and vars, return max posterior and mean + std of vars at max_prob"""
def get_stats(posterior, vars, print_=True, problem=None):
    # Maximum posterior probability
    argmax = np.argmax(posterior)
    prob_max = posterior.flatten()[argmax]
    # Mean and std of each parameter's marginal distribution at prob_max
    unravel_argmax = list(np.unravel_index(argmax, posterior.shape))
    vars_max = [var[unravel_argmax[i]] for i, var in enumerate(vars)]
    probs_mar = marginal_distributions(unravel_argmax, posterior)
    std_mar = [weighted_std(var, prob) for var, prob in zip(vars, probs_mar)]

    return prob_max, vars_max, probs_mar, std_mar

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

""" Reduces the posterior probability of a model to the special null hypothesis
case where one or more of its parameters are set to zero. Returns the max."""
def get_prob_null(posterior, vars, null_dims):
    for dim in sorted(null_dims)[::-1]:
        posterior = np.moveaxis(posterior, dim, -1)
        idx_null = np.argmin(np.abs(vars[dim]))
        posterior = posterior[..., idx_null]
    return posterior.max()

""" Reduces the posterior probability of a model to the special null hypothesis
case where one or more of its parameters are set to zero."""
def get_null_posterior(posterior, vars, null_dims):
    for dim in sorted(null_dims)[::-1]:
        posterior = np.moveaxis(posterior, dim, -1)
        idx_null = np.argmin(np.abs(vars[dim]))
        posterior = posterior[..., idx_null]
    return posterior

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

def export_xy(problem, season, x, y, a, b, std, xlabel="", ylabel=""):
    # write data to txt file
    to_save = pd.DataFrame({"x":x, "y":y})
    save_path = f"data_analyzed/{season}_{problem}_{site}_{YEAR}.txt"
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

def plot_ratio_matrix(site, problem, ratio_matrix):
    fig = plt.figure(figsize=(3.5,3.5))
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 12
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.92, hspace=0.2, wspace=0.2) # adjust the box of axes regarding the figure size
    ax = fig.axes

    # Mask out bayes factor < 10^0.5
    log_ratio_matrix = np.log10(ratio_matrix)
    mask = np.zeros_like(ratio_matrix)
    for i in np.arange(len(mask[0,:])):
        for j in np.arange(len(mask[:,0])):
            if i>j:
                mask[i,j] = True
    log_ratio_matrix[np.abs(log_ratio_matrix) < 0.5] = 0
    if "annotations" in FIGURE_PATH:
        ax = sns.heatmap(log_ratio_matrix, annot=True, vmin=-5, vmax=5, cmap="PRGn",linewidths=0.5, fmt=".1f")
    else:
        ax = sns.heatmap(log_ratio_matrix, annot=False, mask=mask, vmin=-5, vmax=5, cmap="PRGn",linewidths=0.5)

    save_name = f"Ratio_matrix_{problem}_{site}"
    if YEAR:
        save_name += f"_{YEAR}"
    if SEASON:
        save_name += f"_{SEASON}"

    labels = [r"$p_{H_{0}}$",r"$p_{H_{snow}}$",r"$p_{H_{ice}}$",r"$p_{H_{both}}$"]
    ax.set_xticklabels(labels)
    ax.xaxis.tick_top()
    ax.set_yticklabels(labels)
    ax.tick_params(which=u'both',length=0)

    fig.savefig(join(FIGURE_PATH, save_name),transparent=False, dpi=300)
    # plt.show()
    plt.close()

if __name__ == "__main__":
    STEPS = 64  # NOTE Reduce step size to make computations faster.
    band_name = BAND_NAME

    field_list = list()
    snow_slope_list = list()
    snow_K_list = list()
    ice_slope_list = list()
    ice_K_list = list()

    for YEAR in  ["2016", "2017", "2018"]:
        if (YEAR == "2016") & (SEASON != "apr-may"):
            print("Not enough data for all year 2016 or jan 2016")
            # sys.exit()
            continue
        elif (YEAR == "2017") & (SEASON == "jan-feb") & ("orbit13" in DATA_PATH):
            print("No HH data for jan-feb 2017 in orbit 13")
            # sys.exit()
            continue

        for site, data, data_files in get_variables(data_path=DATA_PATH, year=YEAR, season=SEASON, band_name=BAND_NAME):
            # if ("orbit" in data_path) & (site != "D"):
            #     continue
            field_name = site+"_"+YEAR[2:]
            if SEASON == "jan-feb":
                field_name += "01"
            elif SEASON == "apr-may":
                if YEAR == "2018":
                    field_name += "05"
                else:
                    field_name += "04"
            print(field_name)
            field_list.append(field_name)

            print(f"Site {site}")
            log_path = init_log(site, YEAR, SEASON, data_files)
            snow, ice, wind, sig = data[["snow", "ice", "wind", band_name]].values.T


            """Snow"""

            product_dep = np.linspace(-.1, .5, STEPS)
            snow_0 = np.linspace(-1, 1, STEPS)
            snow_noise = np.logspace(-.5, 0.5, STEPS)
            vars = [product_dep, snow_0, snow_noise]
            var_names = [r"$\alpha$", r"$h_{s_{0}}$", r"$\eta_s$"]

            posterior = get_posterior(vars, [wind, np.ones_like(snow)], snow)
            prob_max, vars_max, probs_mar, std_mar = get_stats(
                posterior, vars, problem="Snow",
            )
            prob_null = get_null_posterior(posterior, vars, null_dims=[0]).max()
            stats_info(vars_max, std_mar, prob_max, prob_null,
                site=site, problem="Snow", log_path=log_path)
            save_param(vars_max, std_mar, var_names, site, problem="Snow")

            plot_linear_dependency(
                "snow",
                wind,
                snow,
                *vars_max,
                xlabel="Distance from shore",
                ylabel="Snow thickness",
            )
            export_xy(
                "snow",
                SEASON,
                wind,
                snow,
                *vars_max,
                xlabel="Distance from shore",
                ylabel="Snow thickness",
            )
            plot_parameters(site, "snow", vars, var_names, probs_mar)
            snow_slope_list.append(vars_max[0])
            snow_K_list.append(prob_max / prob_null)

            """Ice"""

            product_dep = np.linspace(-.1, .5, STEPS)
            ice_0 = np.linspace(-1, 1, STEPS)
            ice_noise = np.logspace(-.5, 0.5, STEPS)
            vars = [product_dep, ice_0, ice_noise]
            var_names = [r"$\alpha$", r"$h_{i_{0}}$", r"$\eta_i$"]

            posterior = get_posterior(vars, [wind, np.ones_like(ice)], ice)
            prob_max, vars_max, probs_mar, std_mar = get_stats(
                posterior, vars, problem="Ice",
            )
            prob_null = get_null_posterior(posterior, vars, null_dims=[0]).max()
            stats_info(vars_max, std_mar, prob_max, prob_null,
                site=site, problem="Ice", log_path=log_path)
            save_param(vars_max, std_mar, var_names, site, problem="Ice")

            plot_linear_dependency(
                "ice",
                wind,
                snow,
                *vars_max,
                xlabel="Distance from shore",
                ylabel="Ice thickness",
            )
            export_xy(
                "ice",
                SEASON,
                wind,
                ice,
                *vars_max,
                xlabel="Distance from shore",
                ylabel="Ice thickness",
            )
            plot_parameters(site, "ice", vars, var_names, probs_mar)

            ice_slope_list.append(vars_max[0])
            ice_K_list.append(prob_max / prob_null)

            """Ice vs Snow"""
            snow_dep = np.linspace(-1.5, 1, STEPS)
            ice_0 = np.linspace(-1, 1, STEPS)
            ice_noise = np.logspace(-.5, 1, STEPS)
            vars = [snow_dep, ice_0, ice_noise]
            var_names = [r"$\beta$", r"$h_{i_{0}}$", r"$\eta_i$"]

            # Max posterior
            posterior = get_posterior(vars, [snow, np.ones_like(ice)], ice)
            prob_max, vars_max, probs_mar, std_mar = get_stats(
                posterior, vars, problem="Ice_vs_Snow",
            )
            prob_null = get_null_posterior(posterior, vars, null_dims=[0]).max()
            stats_info(vars_max, std_mar, prob_max, prob_null,
                site=site, problem="Ice_vs_Snow", log_path=log_path)
            save_param(vars_max, std_mar, var_names, site, problem="Ice_vs_Snow")

            plot_linear_dependency(
                "ice_vs_snow",
                snow,
                ice,
                *vars_max,
                xlabel="Snow thickness",
                ylabel="Ice thickness",
            )
            plot_parameters(site, "ice_vs_snow", vars, var_names, probs_mar)


            """ sig H2, H1A, H1B """
            sig_snow_dep = np.linspace(-2, 1.5, STEPS/2)
            sig_ice_dep = np.linspace(-2, 1.5, STEPS/2)
            sig_0 = np.linspace(-1, 1, STEPS/2)
            sig_noise = np.logspace(-.5, 1, STEPS/2)
            vars = [sig_snow_dep, sig_ice_dep, sig_0, sig_noise]
            var_names = [
                r"$\gamma$",
                r"$\delta$",
                r"$\sigma_{sig~0}$",
                r"$\eta_{\sigma_{sig}}$",
            ]

            # Max posterior for H2
            posterior = get_posterior(
                vars,
                [snow, ice, np.ones_like(sig)],
                sig,
            )
            H2_prob_max, H2_vars_max, H2_probs_mar, H2_std_mar = get_stats(
                    posterior, vars, problem="H_both")
            save_param(H2_vars_max, H2_std_mar, var_names, site, problem="H_both")
            plot_parameters(site, "H_both", vars, var_names, H2_probs_mar)

            # Max posterior for H0:
            H0_posterior = get_null_posterior(posterior, vars, null_dims=[0,1])
            H0_vars = [x for i, x in enumerate(vars) if i >=2]
            H0_var_names = [x for i, x in enumerate(var_names) if i >=2]
            H0_prob_max, H0_vars_max, H0_probs_mar, H0_std_mar = get_stats(
                    H0_posterior, H0_vars, problem="H0")
            save_param(H0_vars_max, H0_std_mar, H0_var_names, site, problem="H0")
            plot_parameters(site, "H0", H0_vars, H0_var_names, H0_probs_mar)

            # Max posterior for H1A:
            H1A_posterior = get_null_posterior(posterior, vars, null_dims=[1])
            H1A_vars = [x for i, x in enumerate(vars) if i !=1]
            H1A_var_names = [x for i, x in enumerate(var_names) if i !=1]
            H1A_prob_max, H1A_vars_max, H1A_probs_mar, H1A_std_mar = get_stats(
                    H1A_posterior, H1A_vars, problem="H_snow")
            save_param(H1A_vars_max, H1A_std_mar, H1A_var_names, site, problem="H_snow")
            plot_parameters(site, "H_snow", H1A_vars, H1A_var_names, H1A_probs_mar)

            # Max posterior for H1B:
            H1B_posterior = get_null_posterior(posterior, vars, null_dims=[0])
            H1B_vars = [x for i, x in enumerate(vars) if i !=0]
            H1B_var_names = [x for i, x in enumerate(var_names) if i !=0]
            H1B_prob_max, H1B_vars_max, H1B_probs_mar, H1B_std_mar = get_stats(
                    H1B_posterior, H1B_vars, problem="H_ice")
            save_param(H1B_vars_max, H1B_std_mar, H1B_var_names, site, problem="H_ice")
            plot_parameters(site, "H_ice", H1B_vars, H1B_var_names, H1B_probs_mar)

            # Save prob ratios and stats
            stats_info(H2_vars_max, H2_std_mar, H2_prob_max, H0_prob_max,
                site=site, problem="H_both_v_H0", log_path=log_path)
            stats_info(H2_vars_max, H2_std_mar, H2_prob_max, H1A_prob_max,
                site=site, problem="H_both_v_H_snow", log_path=log_path)
            stats_info(H2_vars_max, H2_std_mar, H2_prob_max, H1B_prob_max,
                site=site, problem="H_both_v_H_ice", log_path=log_path)

            stats_info(H1A_vars_max, H1A_std_mar, H1A_prob_max, H0_prob_max,
                site=site, problem="H_snow_v_H0", log_path=log_path)
            stats_info(H1A_vars_max, H1A_std_mar, H1A_prob_max, H1B_prob_max,
                site=site, problem="H_snow_v_H_ice", log_path=log_path)

            stats_info(H1B_vars_max, H1B_std_mar, H1B_prob_max, H0_prob_max,
                site=site, problem="H_ice_v_H0", log_path=log_path)
            stats_info(H1B_vars_max, H1B_std_mar, H1B_prob_max, H1A_prob_max,
                site=site, problem="H_ice_v_H_snow", log_path=log_path)

            stats_info(H0_vars_max, H0_std_mar, H0_prob_max, H0_prob_max,
                site=site, problem="H0", log_path=log_path)

            # As matrix
            prob_vector = [H0_prob_max, H1A_prob_max, H1B_prob_max, H2_prob_max]
            ratio_matrix = np.zeros([len(prob_vector), len(prob_vector)])
            for i in np.arange(len(prob_vector)):
                ratio_matrix[i,:] = prob_vector/prob_vector[i]
            plot_ratio_matrix(site, band_name, ratio_matrix)

            # Save matrix
            to_save = pd.DataFrame(ratio_matrix, index=["H0", "Hsnow", "Hice", "Hboth"], columns=["H0", "Hsnow", "Hice", "Hboth"])
            save_name = f"Ratio_matrix_{band_name}_{site}"
            if YEAR:
                save_name += f"_{YEAR}"
            if SEASON:
                save_name += f"_{SEASON}"
            to_save.to_csv(join(FIGURE_PATH, save_name+".csv"))

    # Save slope and K values
    to_save = pd.DataFrame({"snow_slope": snow_slope_list, "snow_K": snow_K_list, "ice_slope": ice_slope_list, "ice_K": ice_K_list}, index=field_list)
    to_save.to_csv(join(FIGURE_PATH, f"across-fjord-gradients.csv"))
