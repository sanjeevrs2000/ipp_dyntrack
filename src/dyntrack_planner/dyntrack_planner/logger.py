import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseArray, Point, Pose
from ros_gz_interfaces.msg import ParamVec
from nav_msgs.msg import OccupancyGrid, GridCells, Odometry
import numpy as np
import tf_transformations
from std_msgs.msg import Header, Float32, Float64
from dyntrack_planner.utils import pose_to_numpy, SyncSubscription
from dyntrack_planner.utils import to_logodds, to_prob, sensor_model, negative_sensor_model, compute_entropy, calc_4points_bezier_path, world_to_grid, BoatVisualization
from dyntrack_planner.params import save_name, mission_time, origin, grid_size, theta, d_max, dir_path
from scipy.signal import convolve2d
import os
import csv
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Wedge

run_name = save_name  # Set your run name here
 
class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')
        self.get_logger().info('Logger node started')

        self.create_subscription(Odometry, '/wamv/sensors/position/ground_truth_odometry', self.odom_callback, 10)
        self.create_subscription(OccupancyGrid, 'gridmap', self.og_callback,10)
        self.create_subscription(Float32, "/avg_detections", self.avg_detections_callback, 10)
        self.create_subscription(PoseArray, '/buoy_positions', self.object_pos_callback, 10)
        
        self.log_timer = self.create_timer(10.0, self.log_callback)
        self.get_logger().info('Logger node initialized')
    
        self.occupancy_grid = None
        self.ground_truth = None
        self.trigger = False
        self.start_time = None
        self.avg_det_list = []
        self.entropy_list = []
        self.pos_mse_list = []
        self.og_mse_list = []
        self.path_followed = []
        
        # self.gaussian_kernel = [
        #     [1/16, 2/16, 1/16],
        #     [2/16, 4/16, 2/16],
        #     [1/16, 2/16, 1/16]
        # ]
        
        self.gaussian_kernel = [[0.0020, 0.0133, 0.0219, 0.0133, 0.0020],
            [0.0133, 0.0888, 0.1468, 0.0888, 0.0133],
            [0.0219, 0.1468, 0.2430, 0.1468, 0.0219],
            [0.0133, 0.0888, 0.1468, 0.0888, 0.0133],
            [0.0020, 0.0133, 0.0219, 0.0133, 0.0020]]

        self.gaussian_kernel = np.array(self.gaussian_kernel)
        
    def log_callback(self):
            
        if self.occupancy_grid is None or self.trigger == False:
            return

        if self.get_time() - self.start_time < 10:
            return
        
        en=0
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                en += compute_entropy(self.occupancy_grid[i, j])       
        avg_entropy = en/ (self.grid_size[0]*self.grid_size[1])
        
        if not hasattr(self, 'avg_det'):
            return

        mse, mse_og = self.compute_mse()

        self.entropy_list.append(avg_entropy)
        self.avg_det_list.append(self.avg_det)
        self.pos_mse_list.append(mse)
        self.og_mse_list.append(mse_og)
        
        if self.get_time() - self.start_time > mission_time:
            self.save_data()
            self.get_logger().info('Data saved successfully')
            
            self.destroy_node()

        return            

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
            self.start_time = self.get_time()
            self.avg_det_list.append(0)
            self.entropy_list.append(1)
            
    def og_callback(self, msg:OccupancyGrid):
        
        if self.trigger == False:
            return
        
        w, h = msg.info.width, msg.info.height
        self.grid_size = [w, h]
        self.resolution = msg.info.resolution
        og = np.array(msg.data, dtype=np.int8).reshape(h,w)
        og = og/100.0
        og[og == -1/100] = 0.5
        self.occupancy_grid = og.copy()
        
    def object_pos_callback(self, msg:PoseArray):

        if self.trigger == False or self.occupancy_grid is None:
            return
        
        # Extract positions of buoys
        self.buoy_positions = []
        for pose in msg.poses:
            x,y = pose.position.x, pose.position.y
            if x>origin[0] and x<origin[0]+self.grid_size[0] and y>origin[1] and y<origin[1]+self.grid_size[1]:
                self.buoy_positions.append([pose.position.x, pose.position.y])

        self.ground_truth = np.zeros((self.grid_size[0], self.grid_size[1]))
        for buoy in self.buoy_positions:
            i, j = world_to_grid(buoy[0], buoy[1])
            self.ground_truth[j, i] = 1

    def get_time(self):
        return self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec*1e-9

    def avg_detections_callback(self, msg:Float32):
        self.avg_det = msg.data

    def compute_mse(self):

        if self.occupancy_grid is None or self.ground_truth is None:
            return None

        og = self.occupancy_grid.copy()
        og[og <= 0.5] = 0

        filtered_gt = convolve2d(self.ground_truth, self.gaussian_kernel, mode='same', boundary='wrap', fillvalue=0)
        filtered_og = convolve2d(og, self.gaussian_kernel, mode='same', boundary='wrap', fillvalue=0)

        mse = np.sum((filtered_gt - filtered_og)**2)
        mse_og = np.sum((self.occupancy_grid - filtered_gt)**2) / (self.grid_size[0] * self.grid_size[1])


        # if not hasattr(self, 'boat'):
        #     self.boat = BoatVisualization()

        # # Include the time in the save_name for each plot
        # current_time = self.get_time() - self.start_time
        # time_suffix = f'_t{int(current_time)}'
        # plot_dir = os.path.join(dir_path, 'logs')  #save_dir for plots
        # plot_dir = os.path.join(plot_dir, run_name)
        # os.makedirs(plot_dir, exist_ok=True)

        # path = np.array(self.path_followed) - np.array(origin)
        # # Plot and save the occupancy grid as a separate image
        # fig, ax = plt.subplots(figsize=(11, 8))
        # norm = mcolors.Normalize(vmin=0, vmax=1)
        # ax.imshow(self.occupancy_grid, cmap='viridis', origin='lower', norm=norm)
        # cbar = fig.colorbar(ax.images[0], ax=ax, label='Occupancy Probability')
        # cbar.ax.tick_params(labelsize=20)  # Increase font size for colorbar labels
        # cbar.set_label('Occupancy Probability', fontsize=20)  # Increase font size for the colorbar title
        # ax.plot(path[:, 0], path[:, 1], color='coral', linewidth=3, label='Path Followed')
        # wedge = Wedge((self.pos.x - origin[0], self.pos.y - origin[1]), d_max, 
        #             np.degrees(self.psi - theta), 
        #             np.degrees(self.psi + theta), 
        #             alpha=0.25, color='red', label='FOV')
        # ax.add_patch(wedge)
        # self.boat.plot_boat(ax, self.pos.x - origin[0], self.pos.y - origin[1], self.psi, color='cyan', scale=5)
        # # ax.set_title('Occupancy Grid', fontsize=14)
        # ax.grid(True, color='gray', linestyle='--', alpha=0.7)
        # ax.set_xlim(0, self.grid_size[0])
        # ax.set_ylim(0, self.grid_size[1])
        # ax.set_xlabel('X (m)', fontsize=20)
        # ax.set_ylabel('Y (m)', fontsize=20)
        # ax.tick_params(axis='both', which='major', labelsize=20)  # Increase font size for axis labels
        # plt.tight_layout()
        # plt.savefig(os.path.join(plot_dir, f'{time_suffix}_occupancy_grid.png'), bbox_inches='tight', dpi=300)
        # plt.close()

        # # Plot and save the ground truth as a separate image
        # fig, ax = plt.subplots(figsize=(11, 8))
        # ax.imshow(self.ground_truth, cmap='viridis', origin='lower', norm=norm)
        # # ax.set_title('Ground Truth', fontsize=14)
        # cbar = fig.colorbar(ax.images[0], ax=ax, label='Occupancy Probability')
        # cbar.ax.tick_params(labelsize=16)  # Increase font size for colorbar labels
        # cbar.set_label('Occupancy Probability', fontsize=20)  # Increase font size for the colorbar title
        # ax.grid(True, color='gray', linestyle='--', alpha=0.7)
        # ax.set_xlabel('X (m)', fontsize=20)
        # ax.set_ylabel('Y (m)', fontsize=20)
        # ax.tick_params(axis='both', which='major', labelsize=20)  # Increase font size for axis labels
        # plt.tight_layout()
        # plt.savefig(os.path.join(plot_dir, f'{time_suffix}_ground_truth.png'), bbox_inches='tight', dpi=300)
        # plt.close()

        return mse, mse_og

    def compute_dist_error(self):

        # calculate distance error between estimated positions and ground truth
        if self.ground_truth is None or self.occupancy_grid is None:
            return None
        og = self.occupancy_grid.copy()
        og[og <= 0.5] = 0
        
        # get index of non-zero elements in og
        og_indices = np.argwhere(og > 0.5)
        for ind in og_indices:
            i, j = ind
            # convert to world coordinates
            x, y = world_to_grid(i, j)
            distances = []
            for buoy in self.buoy_positions:
                dist = np.sqrt((x - buoy[0])**2 + (y - buoy[1])**2)
                distances.append(dist)
            avg_distance = np.mean(distances) if distances else None
            return avg_distance
                
    def mse_t0(self):

        og = np.full((self.grid_size[0], self.grid_size[1]), 0.5)

        mse = np.sum((og - self.ground_truth)**2) / (self.grid_size[0] * self.grid_size[1])
        mse_og = np.sum((og - self.ground_truth)**2) / (self.grid_size[0] * self.grid_size[1])

    def save_data(self):
        
        # Create directory if it doesn't exist
        data_dir = os.path.join(dir_path, 'logs')  # save_dir for logs
        data_dir = os.path.join(data_dir, run_name)
        os.makedirs(data_dir, exist_ok=True)
        
        # Set file names based on run_name
        self.det_file = os.path.join(data_dir, f"{run_name}_avgdet_data.csv")
        self.entropy_file = os.path.join(data_dir, f"{run_name}_entropy_data.csv")

        self.pos_mse_file = os.path.join(data_dir, f"{run_name}_pos_mse_data.csv")
        self.og_mse_file = os.path.join(data_dir, f"{run_name}_og_mse_data.csv")

        # Create files with headers if they don't exist

        if not os.path.exists(self.det_file):
            with open(self.det_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['avg_detections'])

        if not os.path.exists(self.entropy_file):
            with open(self.entropy_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['entropy'])

        if not os.path.exists(self.pos_mse_file):
            with open(self.pos_mse_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['pos_mse'])
                
        if not os.path.exists(self.og_mse_file):
            with open(self.og_mse_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['og_mse'])
        
        # Write data to files

        with open(self.det_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.avg_det_list)

        with open(self.entropy_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.entropy_list)

        with open(self.pos_mse_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.pos_mse_list)
            
        with open(self.og_mse_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.og_mse_list)
    
def main():
    rclpy.init()
    logger_node = LoggerNode()
    rclpy.spin(logger_node)
    logger_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()