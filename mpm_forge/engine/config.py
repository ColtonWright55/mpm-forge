from dataclasses import dataclass


@dataclass
class SimConfig:
    n_envs: int = 8
    dt: float = 1e-6
    steps: int = 1000
    n_particles: int = 10000
    grid_res: tuple[int, int, int] = (64, 64, 4)
    use_sparse_grid: bool = False

    E: float = 1e4  # Young's modulus
    nu: float = 0.3  # Poisson's ratio
    yield_stress: float = 1e2
    material_density: float = 1000.0

    is_2d: bool = True
