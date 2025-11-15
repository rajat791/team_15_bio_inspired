import numpy as np
import sys
from capyle.ca import Grid2D, CAConfig

# ---------------------------------------------------------
# TERRAIN PARAMETERS
# ---------------------------------------------------------

# Terrain codes:
# 0 = lake (non burnable)
# 1 = dense forest
# 2 = chaparral
# 3 = canyon
# 4 = town

terrain_ignite_probs = {
    1: 0.2,   # dense forest does not ignite easily
    2: 0.70,   # chaparral ignites easily
    3: 0.90,   # canyon burns easily
    4: 0.75,   # buildings
}

terrain_burn_times = {
    1: 20,     # forest burns slowly
    2: 6,     # chaparral burns several days
    3: 2,    # canyon burns quickly
    4: 4,    # town burns moderately
}


# ---------------------------------------------------------
# SETUP FUNCTION
# ---------------------------------------------------------
def setup(config):
    """
    This function defines:
    - grid size
    - number of generations
    - states used
    - initial conditions
    """

    config.title = "Basic Forest Fire Model"
    config.dimensions = 2

    # State meanings:
    # 0 = unburned
    # 1 = burning
    # 2 = burned out
    config.states = (0, 1, 2, 3)

    # Grid size (you can change later)
    config.grid_dims = (50, 50)

    # Number of timesteps to simulate
    config.num_generations = 200

    # Colors for each state
    config.state_colors = [
        (0.1, 0.6, 0.1),  # unburned - green
        (1, 0, 0),        # burning - red
        (0.2, 0.2, 0.2)   # burned out - dark gray
    ]

    # Create empty grid
    grid = np.zeros(config.grid_dims, dtype=int)

    # Start a fire in the middle
    rows, cols = config.grid_dims
    grid[rows//2, cols//2] = 1  # initial burning cell

    config.initial_grid = grid


# ---------------------------------------------------------
# TRANSITION FUNCTION
# ---------------------------------------------------------
def transition_function(grid, neighbourstates, neighbourcounts):
    """
    Basic rule:
    - burning -> burned out
    - unburned -> burning if ANY neighbour is burning
    - burned out -> stays burned out

    This is intentionally simple to give you a base to extend.
    """

    new_grid = np.copy(grid)

    # Rule 1: burning -> burned out
    burning_cells = (grid == 1)
    new_grid[burning_cells] = 2

    # Rule 2: unburned cells catch fire if ANY neighbour is burning
    # neighbourcounts[1] = number of burning neighbours
    unburned_cells = (grid == 0)
    burning_neighbour = (neighbourcounts[1] > 0)

    ignite = unburned_cells & burning_neighbour
    new_grid[ignite] = 1

    return new_grid


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def main():
    """
    Runs when the model is executed (python fire_model.py).
    CAPyLE calls this automatically when running through the GUI.
    """

    config = CAConfig(sys.argv)
    setup(config)

    # Run CA with simple transition function
    grid = Grid2D(config, transition_function)
    timeline = grid.run()

    return timeline


# Standard Python entry point
if __name__ == "__main__":
    main()
