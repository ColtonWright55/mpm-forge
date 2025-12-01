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
    p_x: wp.array2d(dtype=wp.vec2),
    p_v: wp.array2d(dtype=wp.vec2),
    p_mass: wp.array2d(dtype=float),
    p_stress: wp.array(dtype=wp.mat33),
    p_vol: wp.array(dtype=float),
    grid_mass: wp.array(dtype=float),
    grid_mom: wp.array(dtype=wp.vec3),
    grid_force: wp.array(dtype=wp.vec3),
    # dx: float,
    # inv_dx: float,
):
    """
    Standard particle to grid. Updated Lagrangian specific. This is not what is in Genesis (but similar).
    """
    p = wp.tid()


@wp.kernel
def update_grid(
    grid_m: wp.array4d(dtype=float),
    grid_mv: wp.array4d(dtype=wp.vec3),
    grid_f: wp.array4d(dtype=wp.vec3),
    params: DeviceParams,
    grid_v_next: wp.array4d(dtype=wp.vec3),
):
    p = wp.tid()


@wp.kernel
def g2p(
    grid_v_next: wp.array4d(dtype=wp.vec3),
    params: DeviceParams,
    p_x: wp.array2d(dtype=wp.vec3),
    p_v: wp.array2d(dtype=wp.vec3),
    p_C: wp.array2d(dtype=wp.mat33),
    p_F: wp.array2d(dtype=wp.mat33),
    p_F_plastic: wp.array2d(dtype=wp.mat33),
    p_stress: wp.array2d(dtype=wp.mat33),
    p_vol: wp.array2d(dtype=float),
):
    p = wp.tid()


# while t < t_f:
#     # 1. Reset grid (launch with grid dimensions)
#     wp.launch(kernel=reset_grid, dim=grid_res)

#     # 2. P2G (launch with particle count)
#     wp.launch(kernel=p2g, dim=n_particles, inputs=[...])

#     # 3. Update grid (launch with grid dimensions)
#     # wp.launch(kernel=update_grid, dim=grid_res, inputs=[dt])

#     # 4. G2P (launch with particle count)
#     # wp.launch(kernel=g2p, dim=n_particles, inputs=[dt])

#     t = t + dt


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
        # self.params.dx = config.dx # TODO: ?
        # self.params.inv_dx = 1.0 / config.dx
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
        N = self._n_particles
        H, W = self._grid_res

        # 1. Reset grid
        wp.launch(kernel=reset_grid, dim=(B, H, W), inputs=[self.grid_m, self.grid_v, self.grid_f])

        # 2. P2G
        wp.launch(
            kernel=p2g,
            dim=(self.B, self.N),
            inputs=[
                self.p_x,
                self.p_v,
                self.p_m,
                self.p_vol,
                self.p_C,
                self.p_stress,
                self.params,
                self.grid_m,
                self.grid_mv,
                self.grid_f,
            ],
        )

        # 3. Update grid / apply BCs
        wp.launch(
            kernel=update_grid,
            dim=self.grid_shape,
            inputs=[self.grid_m, self.grid_mv, self.grid_f, self.params, self.grid_v_next],
        )

        # 4. G2P
        wp.launch(
            kernel=g2p,
            dim=(self.B, self.N),
            inputs=[
                self.grid_v_next,
                self.params,
                self.p_x,
                self.p_v,
                self.p_C,
                self.p_F,
                self.p_F_p,
                self.p_stress,
                self.p_vol,
            ],
        )
        # End
        self.t += self.config.dt
