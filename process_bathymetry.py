from os import listdir
from os.path import join

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL.TiffTags import TAGS

DATA_PATH = "data"


def load_bathymetry(file_path):
    with Image.open(file_path) as img:
        bathymetry = np.array(img)
        meta_dict = {
            TAGS[key]: img.tag[key]
            for key in img.tag.keys()
        }

    # bathymetry[bathymetry < -1e+37] = np.nan
    bathymetry[bathymetry < 0] = np.nan
    # for key in meta_dict.keys():
    #     print(key)
    #     print(meta_dict[key])
    # plt.imshow(bathymetry)
    # plt.show()
    return bathymetry, meta_dict


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

    # x, y, x_, y_ = x[:, None], y[:, None], x_[None, :], y_[None, :]
    distances = np.full_like(x, np.inf, dtype=np.float32)
    closest_shore_flat = np.empty([2, x.size])
    print(f"Non NaN points: {x.size}")
    for i, (x_temp, y_temp) in enumerate(zip(x_, y_)):
        if i % 10000 == 0:
            print(f"Iteration {i} on {x_.size}")
        temp_distances = np.sqrt((x_temp-x)**2 + (y_temp-y)**2)
        mask = temp_distances < distances
        distances = np.where(mask, temp_distances, distances)
        if mask.any():
            closest_shore_flat[:, mask] = [[x_temp], [y_temp]]

    distance_to_closest_shore[y, x] = distances
    closest_shore[:, y, x] = closest_shore_flat
    return closest_shore, distance_to_closest_shore


def compute_sections(bathymetry, closest_shore):
    mask = remaining = ~np.isnan(bathymetry)
    sections = np.zeros_like(bathymetry)
    sections[~mask] = np.nan
    # Keep track of the quantity of samples used in averaging sections.
    quantities = np.zeros_like(bathymetry)
    quantities[~mask] = np.nan

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
        y_near, x_near = closest_shore[:, y, x]
        if np.sqrt((y-y_near)**2 + (x-x_near)**2) < 3:
            remaining[y, x] = False
            continue

        if abs(x-x_near) > 1e-8:
            angle = np.arctan((y-y_near)/(x-x_near))
        else:
            angle = np.arctan((y-y_near)/1e-8)
        if (x-x_near) < 0:
            angle += np.pi
        step_y, step_x = .5 * np.sin(angle), .5 * np.cos(angle)
        y_, x_ = np.array([y]), np.array([x])
        y_next, x_next = y_[-1] + step_y, x_[-1] + step_x
        if np.isnan(y_next) or np.isnan(x_next):
            print(y_, y_next, step_y, x_, x_next, step_x, angle)
        while (
                    np.round(y_next).astype(int) < bathymetry.shape[0]
                    and np.round(x_next).astype(int) < bathymetry.shape[1]
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

        y_, x_ = np.round(y_).astype(int), np.round(x_).astype(int)

        segment_length = .5 * len(y_)
        section = segment_length * np.mean(bathymetry[y_, x_])

        x_, y_ = np.unique([x_, y_], axis=1)
        sections[y_, x_] = (
            (quantities[y_, x_]*sections[y_, x_] + section)
            / (quantities[y_, x_] + 1)
        )
        quantities[y_, x_] += 1
        remaining[y, x] = False

    return sections


if __name__ == '__main__':
    for file_path in listdir(DATA_PATH):
        if "bathymetry" in file_path:
            file_path = join(DATA_PATH, file_path)
            bathymetry, meta_dict = load_bathymetry(file_path)
            if "D" in file_path:
                bathymetry = bathymetry[::4, ::4]
            else:
                bathymetry = bathymetry[::2, ::2]
            closest_shore, distance_to_closest_shore = find_closest_shore(
                bathymetry,
            )
            plt.imshow(distance_to_closest_shore)
            plt.show()

            sections = compute_sections(bathymetry, closest_shore)
            plt.imshow(sections)
            plt.show()
