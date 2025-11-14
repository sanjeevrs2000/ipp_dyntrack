#!usr/bin/env python3

import torch
import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose, Twist
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64, Int32
from dyntrack_planner.utils import pose_to_numpy, SyncSubscription
from scipy.ndimage import shift
from dyntrack_planner.utils import to_logodds, to_prob, calc_4points_bezier_path
from dyntrack_planner.utils import batch_get_fov, batch_sensor_model, batch_negative_sensor_model
from dyntrack_planner.params import *
import tensorflow as tf
import os

# set device for torch operations:
if torch.cuda.is_available():
    def_device = 'cuda'
else:
    def_device = 'cpu'

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Error setting memory growth for GPU {gpu}: {e}")


class RHPlannerNode(Node):
    
    def __init__(self):
        
        super().__init__("receding_horizon_planner")
        
        self.wp_publisher = self.create_publisher(PoseArray,'vrx/wayfinding/waypoints', 10)
        self.desired_speed_publisher = self.create_publisher(Float64, 'vrx/wayfinding/desired_speed', 10)

        self.odom_subscriber = self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.og_subscriber = self.create_subscription(OccupancyGrid, 'gridmap', self.og_callback,10)

        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)
        self.occupancy_grid = None

        self.origin = origin
        self.r_tol = r_tol 
        self.x_tol = x_tol
        self.pos = None
        self.wps = PoseArray()
        self.trigger = False        
        self.wind_trigger = False
        self.rt = 0
        self.lt = 0
        self.p_low = p_low
        self.p_high = p_high
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        self.d, self.theta = d_max, theta  # field of view distance and angle
        # self.desired_speed = u
        self.desired_speed = u
        self.l_low, self.l_high = to_logodds(p_low), to_logodds(p_high)

        # self.log_timer = self.create_timer(10, self.log_callback)
        self.planner_timer = self.create_timer(0.1, self.planner)
        self.wp_timer = self.create_timer(0.1, self.wp_callback)
                       
        self.path_followed = []

        # self.model = None
        pth = os.path.join(dir_path, 'models/pred_unet_best')
        self.model = tf.keras.models.load_model(pth)
        # self.model = tf.saved_model.load(pth)
        
        # hyperparameters for cost function
        self.w_coeff = coeff
        # self.w_coeff = 'adaptive'
        if self.w_coeff == "adaptive":
            self.w_coeff = 5
 
        # for receding horizon planner
        self.heading_list = torch.linspace(-0.75*torch.pi, 0.75*torch.pi, steps=7)
        # self.speed_list = torch.linspace(0.5, 2.0, steps=4)
        self.speed_list = torch.tensor(self.desired_speed).unsqueeze(0)
        
        self.action_set = torch.cartesian_prod(self.speed_list, self.heading_list).to(def_device)
            
        self.counter = 1 # for plotting

        self.t_step = 1 #time_step for planner
        self.T = t_ # planning horizon

        self.t_delay = 0 # time delay for prediction for receding horizon planner
        
    def odom_callback(self, msg):
            
        self.pos = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        q=[orientation_q.x,orientation_q.y,orientation_q.z,orientation_q.w]
        self.angular_pos = tf_transformations.euler_from_quaternion(q)
        self.psi= self.angular_pos[2]
        self.linear_vel = msg.twist.twist.linear
        self.angular_vel = msg.twist.twist.angular        
        self.r = self.angular_vel.z
        self.trans, self.rot = pose_to_numpy(msg.pose.pose)
        
        self.path_followed.append([self.pos.x, self.pos.y])
        
        if self.trigger == False:
            init_pos = Pose()
            init_pos.position.x = self.pos.x
            init_pos.position.y = self.pos.y
            init_pos.orientation = orientation_q
            self.wps.poses.append(init_pos)
            self.prev_wp = [self.pos.x, self.pos.y, self.psi]
            self.trigger = True
            self.start_time = self.get_time()
                   
    def wind_callback(self,dir:Float32,speed:Float32):
        self.wind_speed = speed.data
        self.wind_dir = dir.data * np.pi / 180.0  # Convert degrees to radians
        self.wind_trigger = True
    
    def og_callback(self, msg:OccupancyGrid):
        
        if self.trigger == False or self.wind_trigger == False:
            return
        
        w, h = msg.info.width, msg.info.height
        self.grid_size = [w, h]
        self.resolution = msg.info.resolution
        og = np.array(msg.data, dtype=np.int8).reshape(h,w)
        og = og/100.0
        og[og == -1/100] = 0.5
        self.occupancy_grid = og.copy()
        
        x,y = self.wps.poses[-1].position.x, self.wps.poses[-1].position.y
        # dist = np.linalg.norm(np.array([self.pos.x, self.pos.y]) - np.array([x,y]))
        
    def planner(self):
        
        if self.trigger==False or self.occupancy_grid is None:
            return

        x0, y0, psi0 = self.prev_wp[0], self.prev_wp[1], self.prev_wp[2]

        if len(self.wps.poses) < 2:
            pass
        
        else:
            ## replan if current position is 1m before switching condition is met for last waypoint in the current path
            
            xp, yp = self.wps.poses[-2].position.x, self.wps.poses[-2].position.y
            path_vec = (np.array([x0, y0]) - np.array([xp, yp]))/np.linalg.norm(np.array([x0, y0]) - np.array([xp, yp]))

            # computing the cross track error
            proj = np.array([xp, yp]) + np.dot(np.array([self.pos.x, self.pos.y]) - np.array([xp, yp]),path_vec) * path_vec
            along_track_err = np.linalg.norm(np.array([x0, y0]) - proj)

            if np.linalg.norm(np.array([self.pos.x, self.pos.y]) - np.array([x0, y0])) > self.r_tol + 1.0 and along_track_err > self.x_tol + 1.0:
                return

            # approximate time till closest approach to waypoint based on along track error            
            self.t_delay = int(along_track_err / self.desired_speed)
        
        t1 = self.get_time()

        if self.w_coeff == "adaptive":
            time_elapsed = t1 - self.start_time
            self.w_coeff = 5 * (time_elapsed / mission_time)

        ## for mapping - prediction step
        grid = self.occupancy_grid.copy()
        ids = (grid>0.5)
        grid[~ids] = self.p_low
        
        self.pred_grids = self.pred_map(grid, self.T)
        
        # for tracking information gain
        binary_grid = self.occupancy_grid.copy()
        ids = (binary_grid>0.5)
        binary_grid[~ids] = 0
        binary_grid[ids] = 1
        
        self.pred_unc_grids = self.predictions(binary_grid)
        
        # action set        
        states = self.get_trajectories(self.action_set, torch.device(def_device))
        
        utility = self.compute_utility(states)
        
        best_idx = torch.argmax(utility).item()
        best_action = self.action_set[best_idx]
        
        # get waypoint command
        v, delta_psi = best_action.cpu().numpy()

        xe = x0 + v * np.cos(psi0 + delta_psi) * self.T
        ye = y0 + v * np.sin(psi0 + delta_psi) * self.T
        psi_e = psi0 + delta_psi
        psi_e = (psi_e + np.pi) % (2 * np.pi) - np.pi

        n = self.T//self.t_step        
        traj, _ = calc_4points_bezier_path(x0, y0, psi0, xe, ye, psi_e,
            offset=3.0, n_points=n)

        # choose waypoints at n/5 and 2n/5
        wp = Pose()
        wp.position.x = traj[n//5, 0]
        wp.position.y = traj[n//5, 1]
        psi = np.arctan2(traj[n//5, 1] - traj[0, 1], traj[n//5, 0] - traj[0, 0])
        q = tf_transformations.quaternion_from_euler(0,0,psi)
        wp.orientation.x = q[0]
        wp.orientation.y = q[1]
        wp.orientation.z = q[2]
        wp.orientation.w = q[3]
        self.wps.poses.append(wp)

        wp = Pose()
        wp.position.x = traj[2*self.T//5, 0]
        wp.position.y = traj[2*self.T//5, 1]
        psi = np.arctan2(traj[2*self.T//5, 1] - traj[self.T//5, 1], traj[2*self.T//5, 0] - traj[self.T//5, 0])
        q = tf_transformations.quaternion_from_euler(0,0,psi)
        wp.orientation.x = q[0]
        wp.orientation.y = q[1]
        wp.orientation.z = q[2]
        wp.orientation.w = q[3]

        self.wps.poses.append(wp)
        self.prev_wp = [wp.position.x, wp.position.y, psi]

        t2 = self.get_time()
        self.get_logger().info(f"Time taken for planning: {t2-t1} seconds")
        self.get_logger().info(f"Planned waypoints: {self.wps.poses[-2].position.x, self.wps.poses[-2].position.y} \n" \
            f"{self.wps.poses[-1].position.x, self.wps.poses[-1].position.y}")
        
        return None

    def wp_callback(self):                   
        self.wp_publisher.publish(self.wps)
        self.desired_speed_publisher.publish(Float64(data=self.desired_speed))  # publish desired speed
        return            

    def get_trajectories(self, action, device):
        
        # get trajectories with Bezier paths
        K = action.shape[0]
        
        v = action[:,0]
        delta_psi = action[:,1]

        x0, y0, psi0 = self.prev_wp[0], self.prev_wp[1], self.prev_wp[2]

        psi = psi0 + delta_psi
        psi = (psi + torch.pi) % (2  * torch.pi) - torch.pi
        
        xt = x0 + v * torch.cos(psi) * self.T
        yt = y0 + v * torch.sin(psi) * self.T

        xt_np = xt.cpu().numpy()
        yt_np = yt.cpu().numpy()
        psi_np = psi.cpu().numpy()
        
        # Batch calculate trajectories
        
        n_points = int(self.T / self.t_step)

        batch_trajs = np.zeros((K, n_points, 2))

        for i in range(K):
            traj, _ = calc_4points_bezier_path(
                x0, y0, psi0, 
                xt_np[i], yt_np[i], psi_np[i],
                offset=3.0, 
                n_points=n_points
            )
            batch_trajs[i] = traj

        batch_trajs = torch.tensor(batch_trajs, device=device, dtype=torch.float32)
        
        return batch_trajs

    def collision_costs(self, states):
        
        # states are K x T x nx
        K = states.shape[0]
        n_points = states.shape[1]
        device = states.device

        xt = states[:, :, 0]
        yt = states[:, :, 1]

        x_min, y_min = self.origin[0], self.origin[1]
        x_max = self.origin[0] + self.grid_size[0]*self.resolution
        y_max = self.origin[1] + self.grid_size[1]*self.resolution
        
        out_bounds = (xt < x_min) | (xt > x_max) | (yt < y_min) | (yt > y_max)
        
        coll_cost = torch.zeros((K,), device=device)

        # weigh the collision cost based on time step where it is out of bound
        n_points = states.shape[1]
        # create time weights (earlier timesteps penalised more)
        time_idx = torch.arange(n_points, device=device).float()  # 0 .. T-1
        time_weights = (n_points - time_idx) / n_points  # shape (T,)
        weighted_penalty = torch.sum(out_bounds.float() * time_weights.unsqueeze(0), dim=1)  # (K,)
        scale_factor = 0.1 # tune scale_factor
        coll_cost += scale_factor * weighted_penalty

        return coll_cost

    def compute_utility(self, states):
        
        # information gain - entropy + tracking ig
        ent, inf_gain = self.simulate_og_map(states)
        coll_cost = self.collision_costs(states)
        
        utility = ent + self.w_coeff * inf_gain - coll_cost
        
        # print("entropies:{}, tracking_gain: {}, collcost: {}".format(ent, inf_gain, coll_cost))
        
        return utility
        
    ## get step costs and save the map prediction
    def simulate_og_map(self, states):
            
        # states: K x T x nx
        K = states.shape[0]
        n_points = states.shape[1]
        device = states.device

        batch_trajs = states[:,:,:2]

        og = torch.tensor(self.occupancy_grid, device=device, dtype=torch.float32)
        og_batch = og.unsqueeze(0).repeat(K, 1, 1)
        ig = torch.zeros((K,), device=device)
        
        for i in range(1, n_points-1):

            x1 = batch_trajs[:, i-1, 0]
            y1 = batch_trajs[:, i-1, 1]
            x2 = batch_trajs[:, i, 0]
            y2 = batch_trajs[:, i, 1]

            psi_d = torch.atan2(y2 - y1, x2 - x1)

            fov_batch = batch_get_fov(x2, y2, psi_d, device)
            
            # simulate og update in batch
            og_batch = self.simulate_map_update(x2, y2, psi_d, fov_batch, og_batch, i, device)

            # compute tracking information gain for each trajectory
            ig_step = self.compute_track_ig(x2, y2, psi_d, fov_batch, i)
            ig += ig_step

        # compute mutual information from occupancy grid
        entropy_maps = - og_batch * torch.log(og_batch) - (1 - og_batch) * torch.log(1 - og_batch)
        pred_mean_entropy = torch.sum(entropy_maps, dim=(-2, -1))/(self.grid_size[0] * self.grid_size[1])
        
        og_ = torch.tensor(self.occupancy_grid, device=device, dtype=torch.float32)
        entropy_map = -og_ * torch.log(og_) - (1 - og_) * torch.log(1 - og_)
        mean_entropy_now = torch.sum(entropy_map)/(self.grid_size[0] * self.grid_size[1])

        diff_entr = mean_entropy_now - pred_mean_entropy

        ig /= (n_points)
        
        return diff_entr, ig

    def simulate_map_update(self, x, y, psi, fov_mask, og_maps, k, device):
        
        K = x.shape[0]
        
        # Apply prediction updates if available
        if k > 0:
            grid_prev = torch.tensor(self.pred_grids[k-1], device=device, dtype=torch.float32)
            grid_curr = torch.tensor(self.pred_grids[k], device=device, dtype=torch.float32)
            diff_mask = (grid_curr - grid_prev) != 0

            # update_vals = self.alpha * grid_curr + self.beta * og_maps
            # og_maps = torch.where(diff_mask.unsqueeze(0), update_vals.unsqueeze(0), og_maps )

            grid_curr_expanded = grid_curr.unsqueeze(0).expand(K, -1, -1)
            update_vals = self.alpha * grid_curr_expanded + self.beta * og_maps
            # Expand diff_mask to match og_maps dimensions
            diff_mask_expanded = diff_mask.unsqueeze(0).expand(K, -1, -1)
            og_maps = torch.where(diff_mask_expanded, update_vals, og_maps)


        # # simulate occupancy grid map step

        og_maps = torch.clamp(og_maps, self.p_low, self.p_high)
        log_odds_grid = torch.log(og_maps / (1 - og_maps))

        # fov_mask = batch_get_fov(x, y, psi, self.d, self.theta)

        i_coords = torch.arange(self.grid_size[0], device=device).float()
        j_coords = torch.arange(self.grid_size[1], device=device).float()

        i_grid, j_grid = torch.meshgrid(i_coords, j_coords)
        
        x_world = (j_grid + 0.5) * self.resolution + self.origin[0]
        y_world = (i_grid + 0.5) * self.resolution + self.origin[1]

        x_world = x_world.unsqueeze(0).expand(K, -1, -1)
        y_world = y_world.unsqueeze(0).expand(K, -1, -1)
        
        dx = x_world - x.view(K, 1, 1)
        dy = y_world - y.view(K, 1, 1)
        distances = torch.sqrt(dx**2 + dy**2)
        
        # apply fov mask to update only cells within fov
        occ_mask = fov_mask & (og_maps > 0.5)
        free_mask = fov_mask & (og_maps <= 0.5)
        
        # apply updates with predictions from inverse sensor model
        if occ_mask.any():
            occ_distances = torch.where(occ_mask, distances, torch.zeros_like(distances))
            p = batch_sensor_model(occ_distances)
            log_odds_grid += torch.where(occ_mask, log_odds_grid + torch.log(p/(1-p)), torch.zeros_like(log_odds_grid))

        if free_mask.any():
            free_distances = torch.where(free_mask, distances, torch.zeros_like(distances))
            p0 = batch_negative_sensor_model(free_distances)
            log_odds_grid += torch.where(free_mask, log_odds_grid + torch.log(p0/(1-p0)), torch.zeros_like(log_odds_grid))

        # convert log odds to probability
        log_odds_grid = torch.clamp(log_odds_grid, self.l_low, self.l_high)
        og_maps = torch.sigmoid(log_odds_grid)

        return og_maps

    def pred_map(self, grid, t):
            
        wind_factor = 0.03
        
        grid_snapshots = []

        Rx, Ry = 0,0
        for i in np.arange(self.t_step,t+self.t_delay,self.t_step):

            # drift += wind_factor * (self.wind_speed + np.random.normal(0, 0.1 * self.wind_speed)) * np.array([np.cos(self.wind_dir), np.sin(self.wind_dir)])* (dt) 

            dx = wind_factor * self.wind_speed  * np.cos(self.wind_dir) * (self.t_step) / self.resolution
            dy = wind_factor * self.wind_speed * np.sin(self.wind_dir) * (self.t_step) / self.resolution
            Rx += dx
            Ry += dy
            shift_x = np.round(Rx)
            shift_y = np.round(Ry)
            Rx -= shift_x
            Ry -= shift_y

            grid = shift(grid, shift=(shift_y, shift_x), order=1, mode='constant', cval=self.p_low)
            grid_snapshots.append(grid.copy())
                
        return grid_snapshots[self.t_delay//self.t_step:]

    def predictions(self, grid):

        vx = self.wind_speed * np.cos(self.wind_dir)
        vy = self.wind_speed * np.sin(self.wind_dir)

        input_grids = np.array([grid.copy() for _ in range(self.T // self.t_step)])
        input_grids = np.expand_dims(input_grids, axis=-1)  # Add channel dimension
        input_params = np.array([[vx, vy, t] for t in range(self.t_step + self.t_delay, self.t_delay + self.T + self.t_step, self.t_step)])
        
        inputs = (tf.convert_to_tensor(input_grids, dtype=tf.float32),
                  tf.convert_to_tensor(input_params, dtype=tf.float32))

        # predictions = self.model.predict(inputs)
        predictions = self.model(inputs, training=False).numpy()
        predictions[predictions < 0.01] = 0

        # return torch tensor
        predictions = torch.tensor(predictions.squeeze(-1), dtype=torch.float32, device=torch.device(def_device))

        return predictions

    def compute_track_ig(self, x, y, psi, fov_mask, i):
        
        K = fov_mask.shape[0]
        
        pred_unc_batch = self.pred_unc_grids[i].unsqueeze(0).expand(K, -1, -1)
        visible_cells = pred_unc_batch * fov_mask.float()
        valid_mask = (visible_cells > 0.01) & (visible_cells < 1.0) & fov_mask
        
        ig_values = torch.where(valid_mask,
                                torch.exp(-2 * visible_cells),
                                torch.zeros_like(visible_cells))

        total_ig = torch.sum(ig_values, dim=(-2, -1))
        fov_count = torch.sum(fov_mask, dim=(-2, -1)).float()

        avg_ig = torch.where(fov_count > 0, total_ig / fov_count, torch.zeros_like(total_ig))
        
        return avg_ig
        
    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

def main(args=None): 
    rclpy.init(args=args)
    node = RHPlannerNode()  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == "__main__": 
    main()