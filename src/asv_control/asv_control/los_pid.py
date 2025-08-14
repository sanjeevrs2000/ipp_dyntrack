#!/usr/bin/env python3

import rclpy 
from rclpy.node import Node 
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Int64, Float64
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
import tf_transformations
import numpy as np
import math
from asv_control.utils import speed_thrust_wamv

class LOS_PID_Node(Node):

    def __init__(self): 
        super().__init__('los_pd_controller')
        self.wp_subscriber = self.create_subscription(PoseArray,'vrx/wayfinding/waypoints',self.wp_callback,10)
        self.speed_subscriber = self.create_subscription(Float64, 'vrx/wayfinding/desired_speed', self.speed_callback, 10)
        self.wamv_pos_subscriber = self.create_subscription(Odometry,'wamv/sensors/position/ground_truth_odometry', self. odom_callback, 10)
        # self.wamv_imu_subscriber = self.create_subscription(Imu,'wamv/sensors/imu/imu/data',self.imu_gps_callback,10)
        self.left_thrust_publisher = self.create_publisher(Float64, 'wamv/thrusters/left/thrust',10)
        self.right_thrust_publisher = self.create_publisher(Float64, 'wamv/thrusters/right/thrust',10)
        self.left_thruster_pos_publisher = self.create_publisher(Float64, 'wamv/thrusters/left/pos',10)
        self.right_thruster_pos_publisher = self.create_publisher(Float64, 'wamv/thrusters/right/pos',10)

        self.psi = 0
        self.psi_error = 0
        self.r = 0
        self.l_d = 10
        self.r_tol = 3
        self.wp_idx = 0
        self.wps = []
        self.X = 275 # initial value for forward thrust in N
        
        self.trigger = False
        self.wp_trigger = False
        self.timer_= self.create_timer(0.1,self.los_pid_thrust_command)
        self.get_logger().info('LOS_PD_Controller publishing: ')

    def odom_callback(self,msg):
        
        self.pos = msg.pose.pose.position
        self.cur_pos = np.array([self.pos.x, self.pos.y])
        orientation_q = msg.pose.pose.orientation
        q = [orientation_q.x,orientation_q.y,orientation_q.z,orientation_q.w]
        self.angular_pos = tf_transformations.euler_from_quaternion(q)
        self.psi = self.angular_pos[2]
        
        self.linear_vel = msg.twist.twist.linear
        self.angular_vel = msg.twist.twist.angular
        
        self.r = self.angular_vel.z
        self.trigger = True
        
    def wp_callback(self,msg):
        
        if len(msg.poses) == 0:
            self.get_logger().info('No waypoints published')
            return
        
        if len(msg.poses) > len(self.wps):
            
            for i in range(len(self.wps), len(msg.poses)):
                self.wps.append(np.array([msg.poses[i].position.x, msg.poses[i].position.y]))
                
        # for pose in msg.poses:
        #     angles = tf_transformations.euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
        #     self.wps.append(np.array([pose.position.x, pose.position.y]))
        
        self.wp_trigger = True
            
    def speed_callback(self, msg):
        
        speed = msg.data
        
        self.X = speed_thrust_wamv(speed)
        
    def los_pid_thrust_command(self):
        
        if not self.wp_trigger or not self.trigger:
            return
        
        if self.wp_idx >= len(self.wps)-1:
                
            msg=Float64()
            msg.data=0.0
            self.left_thruster_pos_publisher.publish(msg)
            self.right_thruster_pos_publisher.publish(msg)
            self.left_thrust_publisher.publish(msg)
            self.right_thrust_publisher.publish(msg)

            return
                                
        path_vec = (self.wps[self.wp_idx+1] - self.wps[self.wp_idx])/np.linalg.norm(self.wps[self.wp_idx+1]-self.wps[self.wp_idx])   

        # path_vec = (self.wps[1] - self.wps[0])/np.linalg.norm(self.wps[1]-self.wps[0])
        
        # computing the cross track error
        proj = self.wps[self.wp_idx] + np.dot(self.cur_pos-self.wps[self.wp_idx],path_vec) * path_vec
        cross_track = self.cur_pos-proj
        along_track = self.wps[self.wp_idx+1] - proj
        
        if np.linalg.norm(self.cur_pos- self.wps[self.wp_idx+1]) < self.r_tol or np.linalg.norm(along_track) < 0.5:
                
            if self.wp_idx < len(self.wps)-2:
                self.wp_idx += 1
            else:
                self.wp_idx = len(self.wps)-1
                msg=Float64()
                msg.data=0.0
                self.left_thruster_pos_publisher.publish(msg)
                self.right_thruster_pos_publisher.publish(msg)
                return
               
        # computing line of sight point with lookahead distance l_d
        if np.linalg.norm(cross_track) > self.l_d:
            los_point = proj + self.l_d*path_vec
            
        elif np.linalg.norm(self.cur_pos - self.wps[self.wp_idx+1]) < self.l_d:
            los_point = self.wps[self.wp_idx+1]

        else:
            along_track_dist = np.sqrt(self.l_d**2 - (np.linalg.norm(cross_track))**2)
            los_point = proj + along_track_dist*path_vec

        # psi_d = math.atan2(self.wps[1][1]-self.wps[0][1], self.wps[1][0]-self.wps[0][0])
           
        psi_d = math.atan2(los_point[1]-self.cur_pos[1], los_point[0]-self.cur_pos[0])
        psi_d = ssa(psi_d)
        
        # self.get_logger().info(f'heading: {self.psi}, desired heading: {psi_d}')
        # self.get_logger().info(f'cross_track error: {np.linalg.norm(cross_track)}, lookahead distance: {self.l_d}')
        
        # self.get_logger().info(f'Current position: {self.cur_pos}, Goal position: {self.wps[self.wp_idx+1]}, Distance: {np.linalg.norm(self.cur_pos - self.wps[self.wp_idx+1])}')
        print(f'Current position: {self.cur_pos}, Goal position: {self.wps[self.wp_idx+1]}, Distance: {np.linalg.norm(self.cur_pos - self.wps[self.wp_idx+1])}')
        self.psi_error = ssa(psi_d - self.psi)
    
        m=180
        T=1
        k=T/m
        d=1/k
        wn=1
        zeta=0.8
        
        Kp = m*wn**2-k
        Kd = 2*m*zeta*wn-d
        
        Y = 0
        N = -Kp*self.psi_error - Kd*self.r
        X = np.sqrt(self.X**2 - N**2)
        
        #thrust allocation:
        lp_x = -2.373
        ls_x = -2.373
        lp_y = -1.027
        ls_y = 1.027

        b = ls_y - lp_y
        
        T = np.array([[1,0,1,0],[0,1,0,1],[-lp_y,-lp_x,-ls_y,-ls_x]])
        tau_vec = [X,Y,N]
        
        f = np.dot(np.linalg.pinv(T),np.transpose(tau_vec))
        
        
        t_p = self.X*0.5 + N/b
        t_s = self.X*0.5 - N/b
        
        self.publish_thrust_cmds([t_p,t_s,0.0,0.0])

        # log_msg = 'left thrust: '  + str(t_s) + ', angle: ' + str(del_p) + ', right thrust: ' + str(t_s) + ', angle: ' + str(del_s) 
        # self.get_logger().info(str(f))
    
    def publish_thrust_cmds(self,thrust_vec):
        
        self.left_thrust_publisher.publish(Float64(data=thrust_vec[0]))
        self.right_thrust_publisher.publish(Float64(data=thrust_vec[1]))
        self.left_thruster_pos_publisher.publish(Float64(data=thrust_vec[2]))
        self.right_thruster_pos_publisher.publish(Float64(data=thrust_vec[3]))
        
        return
    
def ssa(angle):
    
    #smallest signed angle to constrain angle in [-pi,pi)
    ss_angle = (angle + math.pi) % (2 * math.pi) - math.pi
    return ss_angle
             
                
def main(args=None):
     
    rclpy.init(args=args) 
    node = LOS_PID_Node()
    rclpy.spin(node) 
    rclpy.shutdown() 

  
if __name__ == "__main__": 
    main()