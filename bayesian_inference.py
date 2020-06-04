import numpy as np
from matplotlib import pyplot as plt

from prepare_variables import get_variables, SITE_WIND
from wind_tables import ANGLES


if __name__ == "__main__":
    for site, data, products in get_variables():
        STEPS = 16

        snow_0 = np.linspace(-3, 3, STEPS)
        product_dep = np.linspace(-3, 3, STEPS)
        angles_prob = SITE_WIND[site]
        snow_noise = np.logspace(-.5, .5, STEPS)
        ice_0 = np.linspace(-3, 3, STEPS)
        snow_dep = np.linspace(-3, 3, STEPS)
        velocity_dep = np.linspace(-3, 3, STEPS)
        ice_noise = np.logspace(-.5, .5, STEPS)

        hypotheses = [
            snow_0, product_dep, angles_prob, snow_noise,
            ice_0, snow_dep, velocity_dep, ice_noise,
        ]
        for i, h in enumerate(hypotheses):
            new_shape = np.ones(9, dtype=int)
            new_shape[i] = STEPS
            h = h.astype(np.float16)
            hypotheses[i] = h.reshape(new_shape)
        (
            snow_0, product_dep, angles_prob, snow_noise,
            ice_0, snow_dep, velocity_dep, ice_noise
        ) = hypotheses

        products = products.T
        products = products.reshape(
            [1, 1, products.shape[0], 1, 1, 1, 1, 1, products.shape[1]]
        )
        products = products.astype(np.float32)

        snow, ice, velocity = data.loc[:, ["snow", "ice", "velocity"]].values.T
        new_shape = np.ones(9, dtype=int)
        new_shape[-1] = len(data)
        snow = snow.reshape(new_shape).astype(np.float32)
        ice = ice.reshape(new_shape).astype(np.float32)
        velocity = velocity.reshape(new_shape).astype(np.float32)

        posterior_snow = (
            np.exp(
                -(snow-snow_0-product_dep*products)**2
                / (2*snow_noise**2)
            )
            / snow_noise
        )
        posterior_snow = angles_prob[..., 0] * np.prod(posterior_snow, axis=-1)

        argmax = np.nanargmax(posterior_snow)
        print(argmax, posterior_snow.flatten()[argmax])

        posterior_ice = (
            np.exp(
                -(ice-ice_0+snow_dep*snow+velocity_dep*velocity)**2
                / (2*ice_noise**2)
            )
            / ice_noise
        )
        posterior_ice = np.prod(posterior_ice, axis=-1)

        argmax = np.nanargmax(posterior_ice)

        posterior_hypotheses = posterior_snow * posterior_ice

        argmax = np.nanargmax(posterior_hypotheses)
        print(argmax, posterior_hypotheses.flatten()[argmax])

        raise
