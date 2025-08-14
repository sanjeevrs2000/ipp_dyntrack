import numpy as np
import os

origin = [-532.0, 162.0] # origin of ASV in simulator
grid_size = [100, 100]  # grid size in cells
resolution = 1.0  # resolution in meters per cell

# mapping parameters
p_low, p_high = 0.15, 0.9 # low and high probabilities for occupancy
alpha, beta = 0.99, 0.05  # dynamic occupancy grid update factor

# field of view parameters
d_max = 35 
theta = 5*np.pi/24 # horizontal fov = 2 * 37.5 = 75 degrees

# parameters for the inverse sensor model
a0, d0 = 0.1, 40
a1, d1= 0.03, 40 # negative sensor model 

# for planning
t_ = 25 # planning horizon in seconds
u = 1.5 # speed of the vehicle in m/s
n_cand = 7 # number of candidate trajectories

# coeff = 2 # coefficient for weighting prediction
# coeff = 5 #
coeff = 'adaptive'  # (5)*(1-t/T)

# base_planner = 'coverage'
# base_planner = 'random'
# base_planner = 'greedy'


# planner = 'receding_horizon'
planner = 'sampling_based'

save_name = 'trial_run'

dir_path = '/vrx_ws/src/dyntrack_planner'
# dir_path = os.path.dirname(__file__)

mission_time = 250
