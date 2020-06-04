import numpy as np

ANGLES = np.arange(0, 2*np.pi, np.pi/8)

K_WIND = np.array(
    [.2, .0, .0, .2, 4.4, 3.1, .8, 1.5, 4.7, 1.4, .5, .2, .2, .1, .2, .0]
)

D_WIND = np.array(
    [.0, .0, .0, 13, .0, .0, .0, .0, .0, 8.5, 34, 8.5, .0, .0, .0, .0]
)
D_WIND = D_WIND / 10  # To make winds the same magnitude.

S_WIND = np.array(
    [.0, .0, .1, .1, .0, .0, .0, .0, .1, 2.2, 4.3, 3.8, 2.5, .8, .2, .0]
)
