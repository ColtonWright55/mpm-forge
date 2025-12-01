"""
Simple warp implementation of the first explicit Updated Lagrangian Point Material Point Method (ULMPM) given in `https://doi.org/10.1007/978-3-031-24070-6`
2.5 Algorithm 1.
"""

import warp as wp

from mpm_forge.engine.config import SimConfig


@wp.struct
class DeviceParams:
    dt: float
    dx: float
    inv_dx: float

    is_2d: int

    # Material
    E: float
    nu: float
    mu: float
    lam: float
    yield_stress: float
    hardening: float
    alpha: float


@wp.kernel
def reset_grid(grid_m: wp.array4d(dtype=float), grid_mv: wp.array4d(dtype=wp.vec3), grid_f: wp.array4d(dtype=wp.vec3)):
    env_id, i, j, k = wp.tid()
    grid_m[env_id, i, j, k] = 0.0
    grid_mv[env_id, i, j, k] = wp.vec3(0.0)
    grid_f[env_id, i, j, k] = wp.vec3(0.0)


@wp.kernel
def p2g(
    # Input parameters
    p_x: wp.array2d(dtype=wp.vec3),
    p_v: wp.array2d(dtype=wp.vec3),
    p_m: wp.array2d(dtype=float),
    p_vol: wp.array2d(dtype=float),
    p_stress: wp.array2d(dtype=wp.mat33),
    grid_m: wp.array4d(dtype=float),
    grid_mv: wp.array4d(dtype=wp.vec3),
    grid_f: wp.array4d(dtype=wp.vec3),
    params: DeviceParams,
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
    vp = p_v[b, p]
    mp = p_m[b, p]
    vol = p_vol[b, p]
    stress = p_stress[b, p]

    # Figure out what grid particle is in
    base_pos = xp * params.inv_dx - wp.vec3(1.0)
    v_fx = wp.floor(base_pos[0])
    v_fy = wp.floor(base_pos[1])
    v_fz = wp.floor(base_pos[2])
    base_idx_x = int(v_fx)
    base_idx_y = int(v_fy)
    base_idx_z = int(v_fz)
    grid_pos_floor = wp.vec3(v_fx, v_fy, v_fz)
    fx = base_pos - grid_pos_floor

    # Cubic weights
    wx = cubic_weights(fx[0])
    wy = cubic_weights(fx[1])
    wz = cubic_weights(fx[2])
    dwx = cubic_grad_weights(fx[0], params.inv_dx)
    dwy = cubic_grad_weights(fx[1], params.inv_dx)
    dwz = cubic_grad_weights(fx[2], params.inv_dx)

    # We are using cubic B-splines, and therefore in each direction there are 4 non-zero basis functions. We must add
    # this particles contribution to all its neighboring grid nodes.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ix = base_idx_x + i
                iy = base_idx_y + j
                iz = base_idx_z + k

                # Tensor Product Weight
                w_val = wx[i] * wy[j] * wz[k]

                # Grad (Chain Rule)
                grad_val = wp.vec3(dwx[i] * wy[j] * wz[k], wx[i] * dwy[j] * wz[k], wx[i] * wy[j] * dwz[k])

                # Compute nodal mass
                mass_contrib = mp * w_val
                # wp.atomic_add(grid_m, (b, ix, iy, iz), mass_contrib)

                # Compute nodal momentum
                mom_contrib = mass_contrib * vp
                if params.is_2d == 1:
                    mom_contrib = wp.vec3(mom_contrib[0], mom_contrib[1], 0.0)
                # wp.atomic_add(grid_mv, (b, ix, iy, iz), mom_contrib)

                # Compute external force, internal force, nodal force
                # f_ext_contrib = 0.0 # You could add gravity in here, uninteresting
                # f_int_contrib = -1.0 * vol * (stress * grad_val)
                # f_total_contrib = f_ext_contrib + f_int_contrib
                # wp.atomic_add(grid_f,  (b, ix, iy, iz), f_total_contrib)
                f_contrib = -1.0 * vol * (stress * grad_val)
                if params.is_2d == 1:
                    f_contrib = wp.vec3(f_contrib[0], f_contrib[1], 0.0)
                # wp.atomic_add(grid_f, (b, ix, iy, iz), f_contrib)

    if params.is_2d == 1:
        pass


@wp.func
def cubic_weights(x: float):
    """
    Returns the 4 weights for a cubic B-spline at relative distance x.
    x is the fraction (0.0 to 1.0) of the particle position relative to the cell.

    See [https://doi.org/10.1007/978-3-031-24070-6] Section  3.2
    """

    # Precompute powers
    x2 = x * x
    x3 = x2 * x

    w0 = (1.0 / 6.0) * (1.0 - x) * (1.0 - x) * (1.0 - x)
    w1 = (1.0 / 6.0) * (3.0 * x3 - 6.0 * x2 + 4.0)
    w2 = (1.0 / 6.0) * (-3.0 * x3 + 3.0 * x2 + 3.0 * x + 1.0)
    w3 = (1.0 / 6.0) * x3

    return wp.vec4(w0, w1, w2, w3)


@wp.func
def cubic_grad_weights(x: float, inv_dx: float):
    """
    Returns the 4 gradients (dW/dx) for a cubic B-spline.
    """
    x2 = x * x

    dw0 = (1.0 / 6.0) * (-3.0 * (1.0 - x) * (1.0 - x))
    dw1 = (1.0 / 6.0) * (9.0 * x2 - 12.0 * x)
    dw2 = (1.0 / 6.0) * (-9.0 * x2 + 6.0 * x + 3.0)
    dw3 = (1.0 / 6.0) * (3.0 * x2)

    return wp.vec4(dw0, dw1, dw2, dw3) * inv_dx


# @wp.kernel
# def update_grid(
#     grid_m: wp.array4d(dtype=float),
#     grid_mv: wp.array4d(dtype=wp.vec3),
#     grid_f: wp.array4d(dtype=wp.vec3),
#     params: DeviceParams,
#     grid_v_next: wp.array4d(dtype=wp.vec3),
# ):
#     p = wp.tid()


# @wp.kernel
# def g2p(
#     grid_v_next: wp.array4d(dtype=wp.vec3),
#     params: DeviceParams,
#     p_x: wp.array2d(dtype=wp.vec3),
#     p_v: wp.array2d(dtype=wp.vec3),
#     p_C: wp.array2d(dtype=wp.mat33),
#     p_F: wp.array2d(dtype=wp.mat33),
#     p_stress: wp.array2d(dtype=wp.mat33),
#     p_vol: wp.array2d(dtype=float),
# ):
#     p = wp.tid()


class MPMULSolver:
    """
    Implements the Updated Lagrangian MPM algorithm.
    """

    def __init__(self, config: SimConfig):
        # config
        self.config = config
        # self._grid_density = config.grid_density
        # self._particle_size = config.particle_size
        self._n_envs = config.n_envs
        self._grid_res = config.grid_res
        self._n_particles = config.n_particles
        # self._upper_bound = np.array(config.upper_bound)
        # self._lower_bound = np.array(config.lower_bound)
        # self._leaf_block_size = config.leaf_block_size
        self._use_sparse_grid = config.use_sparse_grid
        # self._enable_CPIC = config.enable_CPIC
        self.B = config.n_envs
        self.N = config.n_particles
        self.grid_shape = (self.B, *config.grid_res)

        self.p_x = wp.zeros((self.B, self.N), dtype=wp.vec3)
        self.p_v = wp.zeros((self.B, self.N), dtype=wp.vec3)
        self.p_m = wp.zeros((self.B, self.N), dtype=float)
        self.p_vol = wp.zeros((self.B, self.N), dtype=float)
        self.p_C = wp.zeros((self.B, self.N), dtype=wp.mat33)
        self.p_F = wp.zeros((self.B, self.N), dtype=wp.mat33)
        self.p_F_p = wp.zeros((self.B, self.N), dtype=wp.mat33)
        self.p_stress = wp.zeros((self.B, self.N), dtype=wp.mat33)

        # This has a batch dimension
        self.grid_m = wp.zeros(self.grid_shape, dtype=float)
        self.grid_mv = wp.zeros(self.grid_shape, dtype=wp.vec3)
        self.grid_f = wp.zeros(self.grid_shape, dtype=wp.vec3)
        self.grid_v_next = wp.zeros(self.grid_shape, dtype=wp.vec3)

        # Prepare Device Params
        self.params = DeviceParams()
        self.params.dt = config.dt
        self.params.dx = 1 / (self.grid_shape[1] - 1)
        self.params.inv_dx = 1.0 / self.params.dx
        # self.params.alpha = config.pic_flip_alpha
        self.params.E = config.E
        self.params.nu = config.nu
        self.params.yield_stress = config.yield_stress
        self.t = 0.0

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------
    def step(self):
        """
        Advance by one explicit time step
        """
        B = self._n_envs
        # N = self._n_particles
        _, H, W = self._grid_res

        # 1. Reset grid
        wp.launch(kernel=reset_grid, dim=(B, H, W), inputs=[self.grid_m, self.grid_mv, self.grid_f])

        # 2. P2G
        wp.launch(
            kernel=p2g,
            dim=(self.B, self.N),
            inputs=[
                self.p_x,
                self.p_v,
                self.p_m,
                self.p_vol,
                self.p_stress,
                self.grid_m,
                self.grid_mv,
                self.grid_f,
                self.params,
            ],
        )

        # 3. Update grid / apply BCs
        # wp.launch(
        #     kernel=update_grid,
        #     dim=self.grid_shape,
        #     inputs=[self.grid_m, self.grid_mv, self.grid_f, self.params, self.grid_v_next],
        # )

        # 4. G2P
        # wp.launch(
        #     kernel=g2p,
        #     dim=(self.B, self.N),
        #     inputs=[
        #         self.grid_v_next,
        #         self.params,
        #         self.p_x,
        #         self.p_v,
        #         self.p_C,
        #         self.p_F,
        #         self.p_F_p,
        #         self.p_stress,
        #         self.p_vol,
        #     ],
        # )
        # End
        self.t += self.config.dt
