from os import listdir
from os.path import join
import re

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from process_bathymetry import load_bathymetry

ANGLE_SITE = {
    'S': 3*np.pi/4,
    'D': np.pi/4,
    'K': -np.pi/8,
}

def read_data(data_path, site, year, season=0, band_name=0, normalize=True):
    files = listdir(data_path)
    files = [
        file for file in files
        if '.csv' in file
        and 'noship' not in file
        and 'both' not in file
        and site in file
        and "bathymetry" not in file
        and "distance" not in file
        and year in file
    ]
    if season == "jan-feb":
        files = [
            file for file in files
            if re.match(".*[0-9]{4}(01|02)[0-9]{2}.*", file)
        ]
    elif season == "apr-may":
        files = [
            file for file in files
            if re.match(".*[0-9]{4}(04|05)[0-9]{2}.*", file)
        ]

    data = [
        pd.read_csv(
            join(data_path, file),
            comment='#',
        ).dropna()
        for file in files
    ]

    for i, d in enumerate(data):
        ice = [s for s in d.columns if "ice" in s.lower()]
        snow = [s for s in d.columns if "snow" in s.lower()]
        assert len(ice) == 1 and len(snow) == 1
        ice, snow = ice[0], snow[0]
        d = d[["long", "lat", ice, snow, band_name]]
        d.columns = ["long", "lat", "ice", "snow", band_name]
        if normalize:
            for var in ["ice", "snow", band_name]:
                d.loc[:, var] = (d[var]-d[var].mean()) / d[var].std()
        data[i] = d

    data = pd.concat(data)

    data = data.loc[~np.isnan(data["ice"])]

    return data, files


def correct_transform(site, transform, ind):
    transform = np.array(
        [
            [transform.a, transform.b, transform.c],
            [transform.d, transform.e, transform.f],
            [0, 0, 1],
        ]
    )

    if site == "D":
        clip = np.array(
            [
                [1, 0, 0],
                [0, 1, 60],
                [0, 0, 1],
            ]
        )
        stretch = np.diag([4, 4, 1])
        transform = transform @ stretch @ clip
    else:
        stretch = np.diag([2, 2, 1])
        transform = transform @ stretch

    return transform


def compute_wind_shore_product(site, center, y, x):
    interp_center = np.empty([0, 2], dtype=float)
    for (x_start, y_start), (x_end, y_end) in zip(center[:-1], center[1:]):
        new_points = np.array(
            [
                np.linspace(x_start, x_end, 6)[:-1],
                np.linspace(y_start, y_end, 6)[:-1],
            ]
        )
        interp_center = np.append(interp_center, new_points.T, axis=0)

    y_, x_ = y[:, None], x[:, None]
    x_center, y_center = interp_center.T
    y_center, x_center = y_center[None, :], x_center[None, :]
    distance_to_center = np.sqrt((y_-y_center)**2 + (x_-x_center)**2)
    args = np.argmin(distance_to_center, axis=1)

    x_center, y_center = interp_center[args].T
    distance = np.sqrt((y-y_center)**2 + (x-x_center)**2)
    angle = np.arctan((y_center-y) / (x-x_center))  # y axis is reversed.
    assert ~np.isnan(angle).any()
    angle[(x-x_center) < 0] += np.pi

    angle_wind = ANGLE_SITE[site]
    product = distance * np.cos(angle-angle_wind)

    return product


def get_variables(data_path, year=0, season=0, band_name=0, plot=False):
    files = [
        file for file in listdir(data_path)
        if "bathymetry" in file and file[-3:] == 'tif'
    ]
    for site in ["S", "D", "K"]:
        if (site == "S") & (year == "2016") & (season != "jan-feb"):
            continue
        if ("orbit" in data_path) & (site != "D"):
            continue
        print(site)
        print(year)
        print(season)
        file_path = [file for file in files if (site in file)][0]
        _, transform = load_bathymetry(join(data_path, file_path))
        velocity = np.load(join(data_path, f"{site}_velocity.npy"))
        data, data_files = read_data(data_path, site, year, season, band_name)
        # data = reads_data(site, year, remove_apr=True, remove_jan=False)
        x_data, y_data, ice, snow, sig = [
            col for _, col in data.iteritems()
        ]

        ind = np.array(
            [
                [0, 0, 1],
                [velocity.shape[1]-1, velocity.shape[0]-1, 1],
            ]
        )
        transform = correct_transform(site, transform, ind)
        x_data, y_data, _ = np.linalg.inv(transform) @ np.array(
            [x_data, y_data, np.ones_like(x_data)]
        )

        i, j = np.round(y_data).astype(int), np.round(x_data).astype(int)
        data["velocity"] = velocity[i, j]
        data.loc[np.isnan(data["velocity"]), "velocity"] = 0

        center = np.loadtxt(
            join(data_path, f"{site}_distance_to_shore_raw.csv"),
            skiprows=1,
            delimiter=',',
            usecols=[1, 2],
        )
        wind = compute_wind_shore_product(site, center, y_data, x_data)
        data["wind"] = wind

        # # Remove data points where current velocity is too far from mean
        # velocity_temp = data["velocity"]
        # velocity_mask = (
        #     velocity_temp < velocity_temp.mean() + .1*velocity_temp.std()
        # )
        # data = data.loc[velocity_mask]
        # x_data, y_data = x_data[velocity_mask], y_data[velocity_mask]

        if plot:
            plt.gcf().set_size_inches(8, 8)
            plt.imshow(np.log(velocity))
            plt.arrow(
                x_data.min(),
                y_data.min(),
                5 * np.cos(-ANGLE_SITE[site]),  # y axis is reversed.
                5 * np.sin(-ANGLE_SITE[site]),  # y axis is reversed.
                color='r',
                width=.5,
            )
            plt.scatter(
                x_data,
                y_data,
                c=data["wind"],
                s=20,
                cmap="seismic",
            )
            plt.xlim([x_data.min()-10, x_data.max()+10])
            plt.ylim([y_data.min()-10, y_data.max()+10])
            plt.gca().invert_yaxis()
            plt.savefig(join(FIGURE_PATH, f"{site}_loc"))
            plt.show()

        yield site, data, data_files


if __name__ == "__main__":
    for _ in get_variables(plot=True):
        pass
