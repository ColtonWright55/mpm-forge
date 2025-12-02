import numpy as np
import warp as wp


wp.init()


@wp.kernel
def p2g(
    # Input parameters
    p_x: wp.array2d(dtype=wp.vec3),
    grid_m: wp.array4d(dtype=float),
):
    """
    Standard particle to grid. Updated Lagrangian specific. This is not what is in Genesis (but similar).

    See [https://doi.org/10.1007/978-3-031-24070-6] Section 2.5 Algorithm 1

    TODO: Obviously there is some stuff in there that can be optimized, if statements etc.
    """
    tid = wp.tid()
    n_particles = p_x.shape[1]
    b = tid // n_particles  # Batch index
    p = tid % n_particles  # Particle index within batch
    xp = p_x[b, p]

    # Figure out what grid particle is in
    inv_dx = float((5-1)/1)  # Imagine you have 5 cells which span 0-1 in xyz
    base_pos = xp * inv_dx - wp.vec3(1.0)
    v_fx = wp.floor(base_pos[0])
    v_fy = wp.floor(base_pos[1])
    v_fz = wp.floor(base_pos[2])
    base_idx_x = int(v_fx)
    base_idx_y = int(v_fy)
    base_idx_z = int(v_fz)

    # We are using cubic B-splines, and therefore in each direction there are 4 non-zero basis functions. We must add
    # this particles contribution to all its neighboring grid nodes.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ix = base_idx_x + i
                iy = base_idx_y + j
                iz = base_idx_z + k

                # Compute nodal mass
                mass_contrib = 4.01
                wp.atomic_add(grid_m, (b, ix, iy, iz), mass_contrib)


@wp.kernel
def p2g2(
    # Input parameters
    p_x: wp.array2d(dtype=wp.vec3),
    grid_m: wp.array(dtype=float),
):
    """
    Standard particle to grid. Updated Lagrangian specific. This is not what is in Genesis (but similar).

    See [https://doi.org/10.1007/978-3-031-24070-6] Section 2.5 Algorithm 1

    TODO: Obviously there is some stuff in there that can be optimized, if statements etc.
    """
    tid = wp.tid()
    n_particles = p_x.shape[1]
    b = tid // n_particles  # Batch index
    p = tid % n_particles  # Particle index within batch
    xp = p_x[b, p]

    # Figure out what grid particle is in
    inv_dx = float((5-1)/1)  # Imagine you have 5 cells which span 0-1 in xyz
    base_pos = xp * inv_dx - wp.vec3(1.0)
    v_fx = wp.floor(base_pos[0])
    v_fy = wp.floor(base_pos[1])
    v_fz = wp.floor(base_pos[2])
    base_idx_x = int(v_fx)
    base_idx_y = int(v_fy)
    base_idx_z = int(v_fz)

    # We are using cubic B-splines, and therefore in each direction there are 4 non-zero basis functions. We must add
    # this particles contribution to all its neighboring grid nodes.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ix = base_idx_x + i
                iy = base_idx_y + j
                iz = base_idx_z + k

                # Compute nodal mass
                mass_contrib = 4.01
                wp.atomic_add(grid_m, (b, ix, iy, iz), mass_contrib)

B = 1
N = 217
grid_bounds = ((0,1),(0,1),(0,1))
grid_res =(5, 5, 5)
grid_shape = (B, *grid_res)
g_m = wp.zeros(grid_shape, dtype=float)
g_m2 = g_m.flatten()
# print(g_m2)

x = np.linspace(0, 1, int(np.floor(np.power(N, 1/3))))
y = np.linspace(0, 1, int(np.floor(np.power(N, 1/3))))
z = np.linspace(0, 1, int(np.floor(np.power(N, 1/3))))
xx,yy,zz = np.meshgrid(x,y,z)
# print(x)
p_x = np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
N = p_x.shape[0]
p_x = np.repeat(p_x[np.newaxis, :, :], B, axis=2)
p_x = wp.array2d(p_x, shape=(B, p_x.shape[1]), dtype=wp.vec3)

# print(p_x)

wp.launch(
    kernel=p2g2,
    dim=(B, N),
    inputs=[
        p_x,
        g_m2,
    ],
)
