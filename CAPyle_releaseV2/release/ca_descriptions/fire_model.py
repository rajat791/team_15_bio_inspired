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
    config.states = (0, 1, 2)

    # Grid size (you can change later)
    config.grid_dims = (100, 100)

    # Number of timesteps to simulate
    config.num_generations = 300

    # Colors for each state
    config.state_colors = [
        (0.1, 0.6, 0.1),  # unburned - green
        (1, 0, 0),        # burning - red
        (0.2, 0.2, 0.2)   # burned out - dark gray
    ]

    rows, cols = config.grid_dims
    
    # Create empty grid
    terrain = np.zeros((rows, cols), dtype=int)
    
    # Terrain codes:
    # 0 = lake
    # 1 = chaparral (default background)
    # 2 = dense forest
    # 3 = canyon scrubland (highly flammable)
    # 4 = town
    
    # Fill entire map with chaparral
    terrain[:, :] = 1

    # Helper scaling factor:
    # Each square in the assignment diagram = 2.5km = 5 cells
    sq = 5

    # ---------------------------------------------------------
    # FOREST BLOCKS (dark green in figure)
    # ---------------------------------------------------------

    # Left L-shaped forest
    terrain[sq*4 : sq*12, sq*2 : sq*6] = 2
    terrain[sq*4 : sq*6, sq*6 : sq*10] = 2

    # Lower-left forest block
    terrain[sq*8 : sq*14, sq*2 : sq*12] = 2

    # Right vertical forest strip
    terrain[sq*4 : sq*12, sq*14 : sq*15] = 2

    # ---------------------------------------------------------
    # LAKES (light blue rectangles)
    # ---------------------------------------------------------
    # Upper horizontal lake
    terrain[sq*7 : sq*8, sq*10 : sq*14] = 0

    # Lower horizontal lake
    terrain[sq*12 : sq*13, sq*6 : sq*16] = 0

    # Vertical lake (middle)
    terrain[sq*7 : sq*13, sq*12 : sq*13] = 0

    # ---------------------------------------------------------
    # CANYON (highly flammable scrubland)
    # ---------------------------------------------------------
    terrain[sq*5 : sq*16, sq*16 : sq*17] = 3

    # ---------------------------------------------------------
    # TOWN (small 2.5km × 2.5km black square)
    # ---------------------------------------------------------
    terrain[sq*1 : sq*2, sq*4 : sq*5] = 4

    # ---------------------------------------------------------
    # Ignition points (NO fire placed yet)
    # ---------------------------------------------------------
    # Power plant (top-left X)
    power_plant = (sq*19, sq*1)

    # Incinerator (top-right X)
    incinerator = (sq*19, sq*19)

    # ---------------------------------------------------------
    # FIRE GRID (dynamic)
    # ---------------------------------------------------------
    fire = np.zeros((rows, cols), dtype=int)

    # Start fire at power plant initially
    fire[power_plant] = 1

    # Attach everything to the config
    config.initial_grid = fire
    config.terrain_grid = terrain
    config.power_plant = power_plant
    config.incinerator = incinerator


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
