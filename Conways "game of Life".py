import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from numba import jit

from random import seed
from random import random


def seed_grid(dim):
    return np.random.choice(a=[True, False], size=(dim, dim), p=[0.1, 0.9])

@jit(nopython=True, parallel=True)
def evolve(grid):

    evolve = np.empty((10000, grid[:1].size,grid[:1].size))
    evolve[0] = grid
    for j in range(0,9998):
        for x in range(grid[:1].size):
            for y in range(grid[:1].size):
                alive = 0
                neigbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
                for i in neigbours:
                    xn = i[0] + x
                    yn = i[1] + y
                    if xn < 0:
                        xn = grid[:1].size - 1
                    elif xn > grid[:1].size - 1:
                        xn = 0
                    if yn < 0:
                        yn = grid[:1].size - 1
                    elif yn > grid[:1].size - 1:
                        yn = 0
                    if evolve[j, xn, yn]:
                        alive = alive + 1
                if alive < 2 or alive > 3:
                        evolve[j+1, x, y] = False
                elif (alive == 2 or alive == 3) and evolve[j, x, y] == True:
                        evolve[j+1, x, y] = True
                elif alive == 3 and evolve[j, x, y] == False:
                        evolve[j+1, x, y] = True


    return evolve

dim = 200
fps = 5
nSeconds = 1000


fig = plt.figure( figsize=(dim,dim))
start = seed_grid(dim)
evolution = evolve(seed_grid(dim))
im = plt.imshow(start, interpolation='none', aspect='auto', vmin=0, vmax=1)


def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(evolution[i])

    return [im]

anim = animation.FuncAnimation(fig, animate_func, frames = nSeconds * fps, interval = 1000 / fps)


plt.show()

print('Done!')








