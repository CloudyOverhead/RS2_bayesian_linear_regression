from os import listdir
from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from spatial_detrending import read_data
from process_bathymetry import load_bathymetry
from wind_tables import ANGLES, K_WIND, D_WIND, S_WIND

DATA_PATH = "data"
FIGURE_PATH = "figures"
SITE_WIND = {
    "K": K_WIND,
    "D": D_WIND,
    "S": S_WIND,
}


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

    distance, angle = distance[:, None], angle[:, None]
    angle_wind = ANGLES[None, :]
    product = distance * np.cos(angle-angle_wind)

    return product


def get_variables(plot=False, select_angle=0):
    files = [
        file for file in listdir(DATA_PATH)
        if "bathymetry" in file and file[-3:] == 'tif'
    ]
    for site in ["K", "D", "S"]:
        file_path = [file for file in files if site in file][0]
        _, transform = load_bathymetry(join(DATA_PATH, file_path))
        velocity = np.load(join(DATA_PATH, f"{site}_velocity.npy"))
        data = read_data(site)
        x_data, y_data, ice, snow = [
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
            join(DATA_PATH, f"{site}_distance_to_shore_raw.csv"),
            skiprows=1,
            delimiter=',',
            usecols=[1, 2],
        )
        product = compute_wind_shore_product(site, center, y_data, x_data)

        velocity_temp = data["velocity"]
        velocity_mask = (
            velocity_temp < velocity_temp.mean() + .1*velocity_temp.std()
        )
        data, product = data.loc[velocity_mask], product[velocity_mask]
        x_data, y_data = x_data[velocity_mask], y_data[velocity_mask]

        if plot:
            plt.gcf().set_size_inches(8, 8)
            plt.imshow(np.log(velocity))
            plt.arrow(
                x_data.min(),
                y_data.min(),
                5 * np.cos(-ANGLES[select_angle]),  # y axis is reversed.
                5 * np.sin(-ANGLES[select_angle]),  # y axis is reversed.
                color='r',
                width=.5,
            )
            plt.scatter(
                x_data,
                y_data,
                c=product[:, select_angle],
                s=20,
                cmap="seismic",
            )
            plt.xlim([x_data.min()-10, x_data.max()+10])
            plt.ylim([y_data.min()-10, y_data.max()+10])
            plt.gca().invert_yaxis()
            plt.savefig(join(FIGURE_PATH, f"{site}_loc"))
            plt.show()

        yield site, data, product


if __name__ == "__main__":
    get_variables(plot=True)
