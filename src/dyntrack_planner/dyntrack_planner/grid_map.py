#!usr/bin/env python3

import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32
from dyntrack_planner.utils import pose_to_numpy, SyncSubscription
from dyntrack_planner.params import origin, grid_size, resolution, p_low, p_high, alpha, beta, d_max, theta, save_name, dir_path
from dyntrack_planner.utils import world_to_grid, grid_to_world, to_logodds, to_prob, sensor_model, negative_sensor_model, get_fov, BoatVisualization
import cv2
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib import colors as mcolors
# import imageio
import os

class OccupancyGridNode(Node): 

    def __init__(self): 
        
        super().__init__("occupancy_grid")

        self.create_subscription(ParamVec, '/wamv/yolo/results', self.per_callback, 10)
        self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/gridmap',10)
        
        wind_df = {'/vrx/debug/wind/direction': Float32, '/vrx/debug/wind/speed': Float32}
        self.wind_sub = SyncSubscription(self, wind_df, self.wind_callback,10,1)
        self.wind_trigger = False
        self.grid = OccupancyGrid()
        self.resolution = resolution
        self.grid_size = grid_size
        self.origin = origin # spawn position of vehicle in world

        self.occupancy_grid = np.full((self.grid_size[1], self.grid_size[0]), 0.5, dtype=float)
        self.log_odds = np.full((self.grid_size[1], self.grid_size[0]),0, dtype=float)

        self.d, self.theta = d_max, theta  # field of view distance and angle
        self.p_low, self.p_high = p_low, p_high  # low and high occupancy probabilities to clip
        self.alpha, self.beta = alpha, beta # dynamic occupancy grid update factor
        
        
        self.l_low = to_logodds(self.p_low)
        self.l_high = to_logodds(self.p_high)
              
        # self.timer = self.create_timer(30, self.wp_callback)
        self.trigger = False
        
        self.gridmaps = []
        self.counter = 0
        self.prev_time = 0
        self.time_ = 0
        self.prev_time = 0
        self.path_followed = []
        
        self.Rx = 0
        self.Ry = 0
        # publish average number of detections
        self.n = 0 # count frames
        self.no_detections = 0 # count number of detections
        self.det_pub = self.create_publisher(Float32, '/avg_detections', 10)
        
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
            self.trigger = True
        
    def per_callback(self, msg):
        
        if self.trigger == False:
            return
        
        self.grid.header = Header()
        # self.grid.header.frame_id = '/wamv/wamv/base_link'
        self.grid.header.frame_id = 'map'
        self.grid.header.stamp = self.get_clock().now().to_msg()

        self.grid.info.height = self.grid_size[1]
        self.grid.info.width = self.grid_size[0]
        self.grid.info.resolution = self.resolution
        self.grid.info.origin.position.x = 0.0
        self.grid.info.origin.position.y = 0.0
       
        self.update_map()
        self.log_odds = np.vectorize(to_logodds)(self.occupancy_grid)
        self.log_odds = np.clip(self.log_odds, self.l_low, self.l_high)

        # self.occupancy_grid = self.calc_drift(self.occupancy_grid)
        # self.log_odds = np.vectorize(self.to_logodds)(self.occupancy_grid)
        # update cells in field of view and register new detections
       
        oc_ids_1 = []
        
        self.n += 1
        
        for ob in msg.params:
            pos = ob.value.double_array_value
            conf = ob.value.double_value
            # if ob.name == 'mb_round_buoy_black' or ob.name == 'mb_round_buoy_orange':    
            if ob.value.integer_value == 7 or ob.value.integer_value == 8:
                self.no_detections += 1        
                # world_pos = (self.rot @ cell_p.T).T + self.trans
                id = world_to_grid(pos[0],pos[1])
                dist = np.sqrt( (self.pos.x - pos[0])**2 + (self.pos.y - pos[1])**2 )
                if id and dist>2 and dist<33:
                    idx,idy = id
                    p0 = sensor_model(dist)
                    for i in range(idx-2,idx+3):
                        for j in range(idy-2,idy+3):
                            if i>=0 and i<self.grid_size[1] and j>=0 and j<self.grid_size[0]:
                                p = p0 *np.exp(-((idx-i)**2 + (idy-j)**2)/(2*(0.0012*dist**2)**2))
                                if p<self.p_low: continue
                                self.log_odds[j,i] += np.log(p/(1-p))
                                oc_ids_1.append((j,i))
                                                                                                     
        fov = get_fov(self.pos.x, self.pos.y, self.psi)
                                   
        for id in fov:
            if  ((id[0],id[1]) not in set(oc_ids_1)):
                j,i = id
                xc, yc = grid_to_world(i, j)
                if xc is None or yc is None:
                    continue
                dist = np.sqrt((xc-self.pos.x)**2 + (yc-self.pos.y)**2)              
                p = negative_sensor_model(dist)
                self.log_odds[j,i] += np.log(p/(1-p))

        self.log_odds = np.clip(self.log_odds, self.l_low, self.l_high)
        self.occupancy_grid = np.vectorize(to_prob)(self.log_odds)
        # self.occupancy_grid = np.clip(self.occupancy_grid, self.p_low, self.p_high)
        og = self.occupancy_grid.copy()*100
        og[og == 50.0] = -1
        self.grid.data = og.astype(np.int8).flatten().tolist()
        self.map_publisher.publish(self.grid)
        self.det_pub.publish(Float32(data=self.no_detections/self.n))        

        # if self.counter % 10 == 0:
        #     self.visualize_map(fov)

        self.counter +=1
               
    def wind_callback(self,dir:Float32,speed:Float32):
        self.wind_speed = speed.data
        self.wind_dir = dir.data * np.pi / 180.0  # Convert degrees to radians
        self.wind_trigger = True
              
    def update_map(self):
        
        grid = self.occupancy_grid.copy()
    
        grid[grid<=0.5] = self.p_low
        
        self.prev_time = self.time_
        self.time_ = self.get_time()
        if self.prev_time == 0:
            self.prev_time = self.time_
            
        dt = self.time_ - self.prev_time
        wind_factor = 0.03
        dx = wind_factor * self.wind_speed * np.cos(self.wind_dir) * dt / self.resolution
        dy = wind_factor * self.wind_speed * np.sin(self.wind_dir) * dt / self.resolution
        
        self.Rx += dx
        self.Ry += dy
        # self.get_logger().info(f"dx, dy: {dx}, {dy}")
        shift_x = np.round(self.Rx)
        shift_y = np.round(self.Ry)
        self.Rx -= shift_x
        self.Ry -= shift_y
        
        out = shift(grid, shift=(shift_y, shift_x), order=1, mode='constant', cval=self.p_low)
                
        # self.get_logger().info(f"Shift: {shift_x}, {shift_y}")

        self.occupancy_grid = np.where((out-grid)!=0, out*self.alpha + self.occupancy_grid*self.beta, self.occupancy_grid)
        # # self.occupancy_grid = np.where((out-grid)!= 0, out, self.occupancy_grid)


    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9
        
    
    def visualize_map(self, fov):
        
        # Create the figure and plot the occupancy grid
        if not hasattr(self, 'boat'):
            self.boat = BoatVisualization()
        
        fig, ax = plt.subplots(figsize=(11, 8))
        norm = mcolors.Normalize(vmin=0, vmax=1) 
        im = ax.imshow(self.occupancy_grid, cmap='viridis', origin='lower', norm=norm)
                    # extent=[self.origin[0], self.origin[0] + self.grid_size[0]*self.resolution,
                    #         self.origin[1], self.origin[1] + self.grid_size[1]*self.resolution])
        
        plt.colorbar(im, ax=ax, label='Occupancy Probability')

        save_dir = os.path.join(dir_path, 'logs')  # save_dir for occupancy grid plots
        save_dir = os.path.join(save_dir, save_name)
        os.makedirs(save_dir, exist_ok=True)
               
        self.boat.plot_boat(ax, self.pos.x - self.origin[0], self.pos.y - self.origin[1],self.psi, color='cyan', scale=5)
        # Add FOV as a translucent wedge
        # Create a wedge to represent the FOV
        wedge = Wedge((self.pos.x - self.origin[0], self.pos.y - self.origin[1]), self.d, 
                    np.degrees(self.psi - self.theta), 
                    np.degrees(self.psi + self.theta), 
                    alpha=0.3, color='red', label='FOV')
        ax.add_patch(wedge)
        
        path = np.array(self.path_followed) - np.array(self.origin)
        plt.plot(path[:, 0], path[:, 1], 'red', linewidth=3, label='Path followed')

        # Add grid lines
        ax.grid(True, color='gray', linestyle='--', alpha=0.7)
        
        # Add title and labels
        # ax.set_title(f'Occupancy Grid - Time: {self.get_time():.1f}s')
        # ax.set_title('Occupancy Grid')
        ax.set_xlabel('X (m)', fontsize=16)
        ax.set_ylabel('Y (m)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(loc='upper right')
        
        # Save the figure
        filename = os.path.join(save_dir, f'map_{self.counter:04d}.png')
        plt.savefig(filename, dpi=250, bbox_inches='tight')
        plt.tight_layout()
        # plt.legend()
        # plt.show()
        plt.close()
        
        # Create animation when we have enough frames (every 500 frames)
        # if self.counter > 0 and self.counter % 500 == 0:
        #     # self.create_animation(save_dir)
        #     with imageio.get_writer(os.path.join(save_dir, 'occupancy_grid_animation3.gif'), mode='I', duration=0.5) as writer:
        #         for i in range(0, self.counter, 10):
        #             filename = os.path.join(save_dir, f'map_{i:04d}.png')
        #             image = imageio.imread(filename)
        #             writer.append_data(image)

    
def main(args=None):
    rclpy.init(args=args) 
    node = OccupancyGridNode()  
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__": 
    main()
    