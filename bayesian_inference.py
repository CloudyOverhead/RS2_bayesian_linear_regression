from os import listdir
from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from spatial_detrending import read_data
from process_bathymetry import load_bathymetry

DATA_PATH = "data"
FIGURE_PATH = "figures"


def correct_transform(site, transform, ind):
    transform = np.array(
        [
            [transform.a, transform.b, transform.c],
            [transform.d, transform.e, transform.f],
            [0, 0, 1],
        ]
    )

    unknown_contraction = np.diag([.5, .5, 1])
    unknown_translation = np.array(
        [
            [1, 0, ind[1, 0]/4],
            [0, 1, ind[1, 1]/4],
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
    transform = transform @ unknown_translation @ unknown_contraction

    return transform


if __name__ == "__main__":
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
        x_map, y_map, _ = transform @ ind.T

        dx = (x_map[1]-x_map[0])/2.
        dy = (y_map[1]-y_map[0])/2.
        extent = [x_map[0]-dx, x_map[-1]+dx, y_map[0]-dy, y_map[-1]+dy]
        plt.figure(figsize=(8, 8))
        plt.imshow(np.log(velocity), extent=extent, origin='lower')
        for var, cmap in zip([ice, snow], ["RdBu", "PuOr"]):
            plt.scatter(
                x_data,
                y_data,
                c=var,
                s=20,
                cmap=cmap,
                vmin=-1,
                vmax=1,
            )
        plt.gca().invert_yaxis()
        plt.xlim([x_data.min()-1000, x_data.max()+1000])
        plt.ylim([y_data.min()-1000, y_data.max()+1000])
        plt.savefig(join(FIGURE_PATH, f"{site}_loc"))
        plt.show()
