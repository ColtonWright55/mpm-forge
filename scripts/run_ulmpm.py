import warp as wp

from mpm_forge.engine.config import SimConfig
from mpm_forge.engine.solvers.ulmpm_solver import MPMULSolver


def main():
    wp.init()

    default_config = SimConfig()
    solver = MPMULSolver(default_config)

    n_steps = default_config.steps
    print("Starting Simulation...")
    for _ in range(n_steps):
        solver.step()
    print("Finished.")


if __name__ == "__main__":
    main()
