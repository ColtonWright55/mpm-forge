"""
Simple warp implementation of the first explicit Updated Lagrangian Point Material Point Method (ULMPM) given in `https://doi.org/10.1007/978-3-031-24070-6`
2.5 Algorithm 1.
"""

import numpy as np
import warp as wp

# Grid properties
grid_res = (128, 128)
grid_mass = wp.array2d(dtype=float, shape=grid_res)
grid_velo = wp.array2d(dtype=wp.vec2, shape=grid_res)
grid_force = wp.array2d(dtype=wp.vec2, shape=grid_res)

# Particle properties
n_particles = 10000
p_pos = wp.array(dtype=wp.vec2, shape=n_particles)
p_velo = wp.array(dtype=wp.vec2, shape=n_particles)
p_F = wp.array(dtype=wp.mat22, shape=n_particles)  # Deformation gradient
p_C = wp.array(dtype=wp.mat22, shape=n_particles)  # Affine velocity matrix (for APIC)
p_stress = wp.array(dtype=wp.mat22, shape=n_particles)
p_vol = wp.array(dtype=float, shape=n_particles)
p_mass = wp.array(dtype=float, shape=n_particles)

dt = 0.0001
t = 0.0
t_f = 1.0


@wp.kernel
def reset_grid():
    """
    Iterate over all the grid nodes and reset their 
    """
    # Get 2D grid index
    i, j = wp.tid()

    grid_mass[i, j] = 0.0
    grid_velo[i, j] = wp.vec2(0.0, 0.0)
    grid_force[i, j] = wp.vec2(0.0, 0.0)

@wp.kernel
def p2g():
    """
    Standard particle to grid pass.
    """


while t < t_f:
    # 1. Reset grid (launch with grid dimensions)
    wp.launch(kernel=reset_grid, dim=grid_res)

    # 2. P2G (launch with particle count)
    wp.launch(kernel=p2g, dim=n_particles, inputs=[...])

    # 3. Update grid (launch with grid dimensions)
    # wp.launch(kernel=update_grid, dim=grid_res, inputs=[dt])

    # 4. G2P (launch with particle count)
    # wp.launch(kernel=g2p, dim=n_particles, inputs=[dt])

    t = t + dt
