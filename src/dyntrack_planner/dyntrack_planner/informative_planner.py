#!usr/bin/env python3

import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64
from dyntrack_planner.utils import pose_to_numpy, SyncSubscription
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import shift
from dyntrack_planner.utils import to_logodds, to_prob, calc_4points_bezier_path, compute_entropy, compute_information_gain
from dyntrack_planner.utils import grid_to_world, world_to_grid, get_fov, sensor_model, negative_sensor_model
from dyntrack_planner.params import origin, p_low, p_high, alpha, beta, d_max, theta, t_, u, n_cand, coeff, mission_time, planner, dir_path
import tensorflow as tf
import os

class PlannerNode(Node): 
 
    def __init__(self): 
        
        super().__init__("informative_planner")

        self.wp_publisher = self.create_publisher(PoseArray,'vrx/wayfinding/waypoints', 10)
        self.desired_speed_publisher = self.create_publisher(Float64, 'vrx/wayfinding/desired_speed', 10)
        self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, 'gridmap', self.og_callback,10)

        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)
        self.occupancy_grid = None
        
        self.create_subscription(Float64, "/wamv/thrusters/left/thrust", self.lt_callback, 10)
        self.create_subscription(Float64, "/wamv/thrusters/right/thrust", self.rt_callback, 10)

        self.origin = origin
        self.r_tol = 5.0
        self.wps = PoseArray()
        self.trigger = False        
        self.wind_trigger = False
        self.rt = 0
        self.lt = 0
        self.p_low = p_low
        self.p_high = p_high
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        self.d, self.theta = d_max, theta  # field of view distance and angle
        self.desired_speed = u
        self.l_low, self.l_high = to_logodds(p_low), to_logodds(p_high)
        self.planner_ = planner
        self.timer = self.create_timer(0.1, self.wp_callback)
                       
        self.entropy_list = []
        self.avg_det_list = []
        self.path_followed = []
        
        # self.model = None
        pth = os.path.join(dir_path, 'models/pred_unet_best')
        self.model = tf.keras.models.load_model(pth)
        
    def odom_callback(self, msg):
            
        self.pos = msg.pose.pose.position
        # self.pos = np.array([pos.x, pos.y])
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
            self.trigger = True
            self.start_time = self.get_time()
                
    def lt_callback(self, msg):
        self.lt = msg.data

    def rt_callback(self, msg):
        self.rt = msg.data
    
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
        dist = np.linalg.norm(np.array([self.pos.x, self.pos.y]) - np.array([x,y]))
        
        if self.rt == 0 and self.lt == 0 and dist < self.r_tol:           
            self.planner()
        
    def planner(self):
        
        t1 = self.get_time()
        if self.trigger==False or self.occupancy_grid is None:
            return
                
        interval = np.pi/4
        theta_range = np.linspace(-3*np.pi/4, 3*np.pi/4, int(1.5*np.pi/interval +1)) 
        entr = []
        inf_gains = []
        d = t_ * self.desired_speed
        
        binary_grid = self.occupancy_grid.copy()
        # ids = (grid>0.4) & (grid!=0.5)
        ids = (binary_grid>0.5)
        binary_grid[~ids] = 0
        binary_grid[ids] = 1
        
        trajectories = []
        self.pred_unc_grids = self.predictions(binary_grid)
        # self.pred_unc_grids = self.predictions_without_nn(binary_grid)
        
        if self.wps is None or len(self.wps.poses) == 0:
            xp = self.pos.x
            yp = self.pos.y
        else:
            xp, yp = self.wps.poses[-1].position.x, self.wps.poses[-1].position.y
        
        for theta in theta_range:
            
            psi = self.psi + theta
            xd, yd = self.pos.x + d*np.cos(psi), self.pos.y + d*np.sin(psi)
            if (xd < self.origin[0] or  xd > self.origin[0] + self.grid_size[0]*self.resolution or \
                yd < self.origin[1] or yd > self.origin[1] + self.grid_size[1]*self.resolution):
                entr.append(-100)
                inf_gains.append(0)
                trajectories.append(None)
                continue 
            
            traj, _ = calc_4points_bezier_path(xp, yp, self.psi, xd, yd, psi, offset=3.0, n_points=int(t_))
            trajectories.append(traj)
            
            # simulate posterior og_map at end of candidate trajectory
            diff_ent, inf_gain = self.simulate_og_map(traj)
            entr.append(diff_ent)
            inf_gains.append(inf_gain)
        
        entr, inf_gains = np.array(entr), np.array(inf_gains)
        
        w = coeff
        if coeff == 'adaptive':
            cur_time = self.get_time() - self.start_time
            w = 5 * (1 - cur_time / mission_time)  # adaptive weighting based on time left in mission

        self.get_logger().info(f'entropy list: {entr}, information gain list: {inf_gains}')
        ind = np.argmax(entr + w*inf_gains)
               
        chosen_traj = trajectories[ind]
        
        if chosen_traj is not None:
            if self.planner_ == 'sampling_based':
                t = np.linspace(t_/5, t_, 5)
            elif self.planner_ == 'receding_horizon':
                t = np.linspace(t_/5, 3*t_/5, 5)
            for idx in t:
                wp = Pose()
                x, y = chosen_traj[int(idx-1)]
                wp.position.x = x
                wp.position.y = y
                self.wps.poses.append(wp)

        t2 = self.get_time()
        self.get_logger().info(f"Time taken for planning: {t2-t1} seconds")
            
    def wp_callback(self):                   
        self.wp_publisher.publish(self.wps)
        self.desired_speed_publisher.publish(Float64(data=self.desired_speed))  # publish desired speed
        return            

    def simulate_og_map(self, path):

        # Simulate an occupancy grid map with possible measurements along the trajectory  
        og = self.occupancy_grid.copy()
        
        grid = self.occupancy_grid.copy()
        ids = (grid>0.5)
        grid[~ids] = 0
        
        pred_grids = self.pred_map(grid, 25)
        inf_gain = 0
        
        for i in range(1, len(path)-1):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            psi = np.arctan2(y2 - y1, x2 - x1)
            fov = get_fov(x2, y2, psi)
            og = self.simulate_map_update(x2, y2, psi, fov, og, pred_grids, i)
            if len(fov) == 0:
                inf_gain += 0
            else:
                ig = 0
                for id in fov:
                    d = np.sqrt((id[0] - x2)**2 + (id[1] - y2)**2)
                    p = self.pred_unc_grids[i][id[0], id[1]]
                    ig += compute_information_gain(p, d)
                inf_gain += ig/ len(fov)

        diff_entr = (np.sum(np.vectorize(compute_entropy)(self.occupancy_grid)) - np.sum(np.vectorize(compute_entropy)(og))) / (self.grid_size[0]*self.grid_size[1])
        inf_gain /= len(path)
        print(f'Information gain: {inf_gain}, Difference in entropy: {diff_entr}')
        
        return diff_entr, float(inf_gain)

    def simulate_map_update(self, x, y, psi, fov, og_map, pred_grids, k):

        if k>0:
            grid = pred_grids[k-1]
            grid_out = pred_grids[k]
            og_map = np.where((grid_out-grid)!=0, self.alpha*grid_out + self.beta*og_map, og_map)

        # # simulate occupancy grid map step
        og_map = np.clip(og_map, self.p_low, self.p_high)
        log_odds_grid = np.vectorize(to_logodds)(og_map)
        oc_ids = []

        for id in fov:
            j, i = id[0], id[1]
            if og_map[j,i] > 0.5:
                xi, yi = grid_to_world(i, j)
                dist = np.sqrt((xi-x)**2 + (yi-y)**2)
                p0 = sensor_model(dist)

                for a in range(i-2, i+3):
                    for b in range(j-2, j+3):
                        if a>=0 and a<self.grid_size[0] and b>=0 and b<self.grid_size[1]:
                            p = p0 * np.exp(-((a-i)**2 + (b-j)**2)/(2*(0.0012*dist**2)**2))
                            if p < self.p_low: continue
                            log_odds_grid[b,a] += np.log(p/(1-p))                    
                            oc_ids.append((b,a))    

        for id in fov:
            if ((id[0],id[1]) not in set(oc_ids)):                
                j,i = id
                xi, yi = grid_to_world(i, j)
                dist = np.sqrt((xi-x)**2 + (yi-y)**2)
                p = negative_sensor_model(dist)
                log_odds_grid[j,i] += np.log(p/(1-p)) 
        
        # convert log odds to probability
        log_odds_grid = np.clip(log_odds_grid, self.l_low, self.l_high)
        og_map =  np.vectorize(to_prob)(log_odds_grid)
        
        return og_map

    def pred_map(self, grid, t, dt=1):
            
        wind_factor = 0.03
        
        grid_snapshots = []

        Rx, Ry = 0,0
        for i in np.arange(dt,t,dt):
           
            dx = wind_factor * self.wind_speed  * np.cos(self.wind_dir) * (dt) / self.resolution
            dy = wind_factor * self.wind_speed * np.sin(self.wind_dir) * (dt) / self.resolution
            Rx += dx
            Ry += dy
            shift_x = np.round(Rx)
            shift_y = np.round(Ry)
            Rx -= shift_x
            Ry -= shift_y

            grid = shift(grid, shift=(shift_y, shift_x), order=1, mode='constant', cval=self.p_low)
            grid_snapshots.append(grid.copy())
                
        return grid_snapshots

    def predictions(self, grid):

        if self.model is None:
            path = os.path.join(dir_path, 'models/pred_unet_best') # path to the spatiotemporal network
            self.model = tf.keras.models.load_model(path)

        vx = self.wind_speed * np.cos(self.wind_dir)
        vy = self.wind_speed * np.sin(self.wind_dir)

        input_grids = np.array([grid.copy() for _ in range(t_)])
        input_grids = np.expand_dims(input_grids, axis=-1)  # Add channel dimension
        input_params = np.array([[vx, vy, t] for t in range(1, t_+1)])
        
        inputs = (tf.convert_to_tensor(input_grids, dtype=tf.float32),
                  tf.convert_to_tensor(input_params, dtype=tf.float32))

        predictions = self.model.predict(inputs)
        predictions[predictions < 0.01] = 0

        return predictions
        
    # def predictions_without_nn(self, grid):
        
    #     vx = self.wind_speed * np.cos(self.wind_dir)
    #     vy = self.wind_speed * np.sin(self.wind_dir)
    #     gamma = 0.03
    #     grid_snapshots = []
    #     for t in range(t_):
    #         dx = gamma * vx * t/ self.resolution
    #         dy = gamma * vy * t/ self.resolution

    #         grid_ = shift(grid, shift=(dy, dx), order=1, mode='constant', cval=0)
    #         grid_snapshots.append(grid_.copy())

    #     return grid_snapshots

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9
        

def main(args=None): 
    rclpy.init(args=args)
    node = PlannerNode()  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == "__main__": 
    main()
    
