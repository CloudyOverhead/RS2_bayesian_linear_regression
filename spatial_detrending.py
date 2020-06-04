from os import listdir
from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import lsq_linear

DATA_DIR = "data"


def read_data(site, normalize=True):
    files = listdir(DATA_DIR)
    files = [
        file for file in files
        if '.csv' in file
        and 'transect' not in file
        and site in file
        and "bathymetry" not in file
        and "distance" not in file
    ]

    data = [
        pd.read_csv(
            join(DATA_DIR, file),
            comment='#',
        )
        for file in files
    ]

    for i, d in enumerate(data):
        ice = [s for s in d.columns if "ice" in s.lower()]
        snow = [s for s in d.columns if "snow" in s.lower()]
        assert len(ice) == 1 and len(snow) == 1
        ice, snow = ice[0], snow[0]
        d = d[["long", "lat", ice, snow]]
        d.columns = ["long", "lat", "ice", "snow"]
        if normalize:
            for var in ["ice", "snow"]:
                d.loc[:, var] = (d[var]-d[var].mean()) / d[var].std()
        data[i] = d

    data = pd.concat(data)

    return data


def remove_trend(x, y, values, d_neighbors=.1):
    x, y, values = np.array(x), np.array(y), np.array(values)
    x, y = x[:, None], y[:, None]
    distances = np.sqrt((x-x.T)**2 + (y-y.T)**2)
    d_neighbors *= distances.max()

    neighbors_matrix = distances < d_neighbors

    mask = np.isnan(values)
    neighbors_matrix[mask] = False
    neighbors_matrix[:, mask] = False
    values[mask] = 0

    local_means = neighbors_matrix @ values / neighbors_matrix.sum(axis=1)
    values -= local_means
    values[mask] = np.nan

    return values


# def remove_trend(x, y, values):
#     x, y, values = np.array(x), np.array(y), np.array(values)
#
#     mask = ~np.isnan(values)
#     x_filtered, y_filtered, values_filtered = x[mask], y[mask], values[mask]
#
#     A = np.array([x_filtered, y_filtered, np.ones_like(x_filtered)]).T
#     a, b, c = lsq_linear(A, values_filtered).x
#
#     values -= a*x + b*y + c
#
#     return values

if __name__ == "__main__":
    fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=3)
    for i, var in enumerate(["snow", "ice"]):
        for j, place in enumerate(["S", "D", "K"]):
            d = read_data(place)
            d[var] = remove_trend(d["long"], d["lat"], d[var])
            plt.sca(axes[i, j])
            plt.scatter(
                d["long"],
                d["lat"],
                c=d[var],
                cmap="seismic",
                vmin=-1,
                vmax=1,
            )

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("Snow")
    axes[1, 0].set_ylabel("Ice")
    axes[1, 0].set_xlabel("S")
    axes[1, 1].set_xlabel("D")
    axes[1, 2].set_xlabel("K")
    plt.show()

    fig, axes = plt.subplots(figsize=(15, 5), nrows=1, ncols=3)
    for i, place in enumerate(["S", "D", "K"]):
        d = read_data(place)
        d[var] = remove_trend(d["long"], d["lat"], d[var])
        plt.sca(axes[i])
        plt.scatter(d["snow"], d["ice"], s=4, c='k')

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("Ice")
    axes[0].set_xlabel("Snow")
    plt.show()
