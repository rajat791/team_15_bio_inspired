# Name: Forest Fire Model
# Dimensions: 2

import sys
import inspect
import numpy as np
global config

# --- CAPyle path setup (do not modify) ---
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# -----------------------------------------

from capyle.ca import Grid2D
import capyle.utils as utils

neighbors = [(dr, dc) for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0)]

# Fire spread probability per terrain type
ignite_prob = {
    1: 0.45,   # chaparral
    2: 0.04,   # forest
    3: 0.90,   # canyon
    4: 0.50,   # town
    5: 1.0,
    6: 0.0,
}

# Burn durations
burn_duration = {
    1: 25,
    2: 60,
    3: 10,
    4: 40,
}

# Vegetation regrowth probability per terrain type
veg_spread_prob = {
    1: 0.01,
    2: 0.0002,
    3: 0.01
}

def setup(args):
    global config
    config_path = args[0]
    config = utils.load(config_path)
    config.title = "Forest Fire Model – Assignment Map"
    config.dimensions = 2
    config.states = (0, 1, 2, 3, 4, 5, 6)
    config.grid_dims = (200, 200)
    config.num_generations = 300

    # Colours
    config.state_colors = [
        (0.2, 0.4, 1.0),  # 0 lake
        (0.95, 0.9, 0.3), # 1 chaparral
        (0.0, 0.3, 0.0),  # 2 forest
        (1.0, 1.0, 0.1),  # 3 canyon
        (0.0, 0.0, 0.0),  # 4 town
        (1.0, 0.0, 0.0),  # 5 burning
        (0.4, 0.4, 0.4)   # 6 burned
    ]

    # Wind
    config.wind_direction = "SE"
    config.wind_strength = 5.0
    
    # Wind direction 
    wind_vectors = {
        "N":  (-1,  0),
        "NE": (-1,  1),
        "E":  ( 0,  1),
        "SE": ( 1,  1),
        "S":  ( 1,  0),
        "SW": ( 1, -1),
        "W":  ( 0, -1),
        "NW": (-1, -1),
    }
    
    config.wind_vec = normalise(wind_vectors[config.wind_direction])

    # ============================================================
    # TERRAIN GRID
    # ============================================================
    rows, cols = config.grid_dims
    terrain = np.zeros((rows, cols), dtype=int)
    terrain[:, :] = 1  # default chaparral

    # Town
    terrain[180:190, 55:65] = 4

    # Lakes
    terrain[160:170, 100:160] = 0
    terrain[40:80, 70:80] = 0

    # Canyon
    terrain[40:130, 140:150] = 3

    # Forest blocks
    terrain[20:100, 20:50] = 2
    terrain[100:140, 20:100] = 2
    terrain[20:30, 50:80] = 2
    terrain[100:140, 100:130] = 2
    terrain[20:140, 0:20] = 2
    terrain[140:190, 20:45] = 2
    terrain[140:190, 75:90] = 2
    terrain[140:165,45:75] = 2



    # Power plant & incinerator (flammable chaparral)
    terrain[0:5, 17:22] = 1
    terrain[0:5, 195:200] = 1

    # ============================================================
    # FIRE GRID
    # ============================================================
    fire = np.zeros((rows, cols), dtype=int)
    fire[0:5, 17:22] = 5
    fire[0:5, 195:200] = 5

    # ============================================================
    # BURN TIME GRID
    # ============================================================
    burn_time = np.zeros((rows, cols), dtype=float)
    burn_time[0:5, 17:22] = 3
    burn_time[0:5, 195:200] = 3

    # ============================================================
    # INITIAL GRID FOR GUI (TERRAIN + FIRE OVERLAY)
    # ============================================================
    initial_vis = terrain.copy()
    initial_vis[fire == 5] = 5

    config.initial_grid = initial_vis
    config.terrain_grid = terrain
    config.fire_grid = fire
    config.burn_time_grid = burn_time

    if len(args) == 2:
        config.save()
        sys.exit()

    return config

def normalise(vec):
    x, y = vec
    mag = np.sqrt(x*x + y*y)
    if mag == 0:
        return np.array([0.0, 0.0])
    return np.array([x/mag, y/mag])


def spread_fire(terrain, fire, new_fire, burn_time):
    # Spread fire
    rows, cols = terrain.shape
    
    burning_cells = np.where(fire == 5)
    for br, bc in zip(burning_cells[0], burning_cells[1]):
        for dr, dc in neighbors:
            nr, nc = br + dr, bc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Skip lakes
                if terrain[nr, nc] == 0 or new_fire[nr, nc] in (5, 6):
                    continue

                terr = terrain[nr, nc]

                # Probability catch fire
                spread_prob = ignite_prob.get(terr, 0.1)

                direction_vec = normalise((dr, dc))
                wind_vec = config.wind_vec
                alignment = np.dot(wind_vec, direction_vec)

                if alignment > 0:
                    spread_prob *= (1 + config.wind_strength * alignment)
                elif alignment < 0:
                    spread_prob *= (1 + (alignment*0.6) * (config.wind_strength * 0.4))

                if config.initial_grid[nr, nc] == 3 and config.initial_grid[br, bc] != 3:
                    # Lower the probability of spread if the neighbor is in a canyon and this tile is on the mainland
                    spread_prob *= 0.0002
                elif config.initial_grid[nr, nc] != 3 and config.initial_grid[br, bc] == 3:
                    # Lower the probability of spread if the neighbor is on mainland and this tile is in a canyon
                    spread_prob *= 0.0003
                else:
                    # Humidity is higher near lakes, reducing the fire spread probability
                    humidity_factor = 0
                    for sr, sc in neighbors:
                        r, c = sr + nr, sc + nc
                        if 0 <= r < rows and 0 <= c < cols and terrain[r, c] == 0:
                            humidity_factor += 1
                    if humidity_factor >= 3:
                        spread_prob *= 0.005
                    else:
                        spread_prob /= (1+humidity_factor)

                # Attempt to ignite neighbor
                if np.random.rand() < spread_prob:
                    new_fire[nr, nc] = 5
                    burn_time[nr, nc] = burn_duration.get(terr, 15)

def spread_vegetation(terrain, new_fire, burn_time):
    rows, cols = terrain.shape
    veg_cells = np.where((terrain >= 1) & (terrain <= 3))
    for br, bc in zip(veg_cells[0], veg_cells[1]):
        for dr, dc in neighbors:
            nr, nc = br + dr, bc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Attempt to spread vegetation into the burned area
                if terrain[nr, nc] == 6 and np.random.rand() < veg_spread_prob.get(terrain[br,bc], 0):
                    # Prohibit spread of vegetation across the canyon border
                    moving_into_canyon = terrain[br,bc] != 3 and config.initial_grid[nr, nc] == 3
                    moving_out_canyon = terrain[br,bc] == 3 and config.initial_grid[nr, nc] != 3
                    if moving_into_canyon or moving_out_canyon:
                        continue

                    # Spread the vegetation into the burned area
                    terrain[nr, nc] = terrain[br, bc]
                    new_fire[nr, nc] = 0
                    burn_time[nr, nc] = 0

def transition_function(grid, neighbourstates, neighbourcounts):
    terrain = config.terrain_grid
    burn_time = config.burn_time_grid
    fire = config.fire_grid
    new_fire = fire.copy()

    # Update burning cells
    burning_mask = (fire == 5)
    burn_time[burning_mask] -= 1
    finished = burning_mask & (burn_time <= 0)
    new_fire[finished] = 6  # burnt

    # Update the map
    spread_fire(terrain, fire, new_fire, burn_time)
    spread_vegetation(terrain, new_fire, burn_time)

    # -----------------------------
    # Update visual grid
    # -----------------------------
    out = terrain.copy()
    out[new_fire == 5] = 5
    out[new_fire == 6] = 6
    config.fire_grid = new_fire
    config.terrain_grid = out

    return out

def main():
    global config
    config = setup(sys.argv[1:])
    grid = Grid2D(config, transition_function)
    timeline = grid.run()

    config.save()
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
