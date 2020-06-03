from os import listdir
from os.path import join

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS
import rasterio

DATA_PATH = "data"
FIGURE_PATH = "figures"


def load_bathymetry(file_path):
    dataset = rasterio.open(file_path)
    bathymetry = dataset.read(1)
    bathymetry[bathymetry < 0] = np.nan
    return bathymetry, dataset.transform


def find_closest_shore(bathymetry):
    x, y = np.meshgrid(
        np.arange(bathymetry.shape[1]),
        np.arange(bathymetry.shape[0]),
    )
    distance_to_closest_shore = np.full_like(bathymetry, np.nan)
    closest_shore = np.full([2, *bathymetry.shape], np.nan)

    bathymetry, x, y = bathymetry.flatten(), x.flatten(), y.flatten()
    mask = np.isnan(bathymetry)
    x_, y_ = x[mask], y[mask]
    x, y = x[~mask], y[~mask]

    distances = np.full_like(x, np.inf, dtype=np.float32)
    closest_shore_flat = np.empty([2, x.size])
    for i, (x_temp, y_temp) in enumerate(zip(x_, y_)):
        temp_distances = np.sqrt((x_temp-x)**2 + (y_temp-y)**2)
        mask = temp_distances < distances
        distances = np.where(mask, temp_distances, distances)
        if mask.any():
            closest_shore_flat[:, mask] = [[x_temp], [y_temp]]

    distance_to_closest_shore[y, x] = distances
    closest_shore[:, y, x] = closest_shore_flat
    return closest_shore, distance_to_closest_shore


def compute_sections(bathymetry, picks):
    mask = remaining = ~np.isnan(bathymetry)
    sections = np.zeros_like(bathymetry)
    sections[~mask] = np.nan

    coords = np.meshgrid(
        np.arange(bathymetry.shape[0]),
        np.arange(bathymetry.shape[1]),
        indexing='ij',
    )
    coords = np.moveaxis(coords, 0, -1)
    while remaining.any():
        coords_remaining = coords[remaining]
        ind = np.random.choice(coords_remaining.shape[0])
        y, x = coords_remaining[ind]

        angle = get_angle(picks, y, x)
        angle += np.pi / 2
        step_y, step_x = .5 * np.sin(angle), .5 * np.cos(angle)

        y_, x_ = compute_linepoints(
            bathymetry,
            y,
            step_y,
            x,
            step_x,
        )
        y_, x_ = compute_linepoints(
            bathymetry,
            y_[::-1],
            -step_y,
            x_[::-1],
            -step_x,
        )

        y_, x_ = np.round(y_).astype(int), np.round(x_).astype(int)

        segment_length = .5 * len(y_)
        section = segment_length * np.mean(bathymetry[y_, x_])

        sections[y, x] = section
        remaining[y, x] = False

    return sections


def compute_linepoints(bathymetry, start_y, step_y, start_x, step_x):
    if isinstance(start_y, np.ndarray):
        y_ = start_y
    else:
        y_ = np.array([start_y])
    if isinstance(start_y, np.ndarray):
        x_ = start_x
    else:
        x_ = np.array([start_x])

    y_next, x_next = y_[-1] + step_y, x_[-1] + step_x
    while (
                np.round(y_next).astype(int) < bathymetry.shape[0]
                and np.round(x_next).astype(int) < bathymetry.shape[1]
                and np.round(y_next).astype(int) >= 0
                and np.round(x_next).astype(int) >= 0
                and not np.isnan(
                    bathymetry[
                        np.round(y_next).astype(int),
                        np.round(x_next).astype(int),
                    ]
                )
            ):
        y_ = np.append(y_, y_next)
        x_ = np.append(x_, x_next)
        y_next, x_next = y_next + step_y, x_next + step_x

    return y_, x_


def get_angle(picks, y, x):
    y_picks, x_picks = picks.T
    distances = np.sqrt((x_picks-x)**2 + (y_picks-y)**2)
    closest = np.argmin(distances)
    picks = np.concatenate([[picks[0]], picks, [picks[-1]]], axis=0)
    before, closest, after = picks[[closest, closest+1, closest+2]]
    if (closest[1]-before[1]) > 1e-8:
        angle_before = np.arctan((closest[0]-before[0])/(closest[1]-before[1]))
    else:
        angle_before = np.arctan((closest[0]-before[0])/1e-8)
    if (closest[1]-before[1]) < 0:
        angle_before += np.pi
    if (after[1]-closest[1]) > 1e-8:
        angle_after = np.arctan((after[0]-closest[0])/(after[1]-closest[1]))
    else:
        angle_after = np.arctan((after[0]-closest[0])/1e-8)
    if (after[1]-closest[1]) < 0:
        angle_after += np.pi

    if angle_after > angle_before + np.pi/2:
        angle_before += np.pi
    elif angle_before > angle_after + np.pi/2:
        angle_after += np.pi

    return np.mean([angle_before, angle_after])


if __name__ == '__main__':
    for site in ["K", "D", "S"]:
        file_path = f"{site}_bathymetry_UTM_18N.tif"
        file_path = join(DATA_PATH, file_path)
        bathymetry, _ = load_bathymetry(file_path)
        if site == "D":
            CLIP_Y = 60
            bathymetry = bathymetry[::4, ::4]
            bathymetry = bathymetry[CLIP_Y:]
        else:
            bathymetry = bathymetry[::2, ::2]

        shore, distance_to_shore = find_closest_shore(bathymetry)
        np.save(
            join(DATA_PATH, f"{site}_distance_to_shore"),
            distance_to_shore,
        )
        np.save(
            join(DATA_PATH, f"{site}_shore"),
            shore,
        )
        plt.imshow(distance_to_shore)
        plt.savefig(join(FIGURE_PATH, f"{site}_distance_to_shore"))
        plt.show()

        picks = np.loadtxt(file_path + '.csv', skiprows=1, delimiter=',')
        picks = picks[:, :0:-1]  # (_, x, y) to (i, j)
        if "D" in file_path:
            picks = picks / 4
            picks[:, 0] -= CLIP_Y
            picks = picks[:-3]
        else:
            picks = picks / 2
        plt.imshow(bathymetry)
        plt.scatter(picks[:, 1], picks[:, 0], c='r', s=3)
        plt.savefig(join(FIGURE_PATH, f"{site}_picks"))
        plt.show()
        sections = compute_sections(bathymetry, picks)
        velocity = 1 / sections
        np.save(join(DATA_PATH, f"{site}_velocity"), velocity)
        plt.imshow(np.log(velocity))
        plt.savefig(join(FIGURE_PATH, f"{site}_velocity"))
        plt.show()
