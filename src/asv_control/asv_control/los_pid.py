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
from dyntrack_planner.params import l_d, r_tol, x_tol

class LOS_PID_Node(Node):

    def __init__(self): 
        super().__init__('los_pd_controller')
        self.wp_subscriber = self.create_subscription(PoseArray,'vrx/wayfinding/waypoints',self.wp_callback,10)
        self.speed_subscriber = self.create_subscription(Float64, 'vrx/wayfinding/desired_speed', self.speed_callback, 10)
        self.wamv_pos_subscriber = self.create_subscription(Odometry,'wamv/sensors/position/ground_truth_odometry', self. odom_callback, 10)
        self.left_thrust_publisher = self.create_publisher(Float64, 'wamv/thrusters/left/thrust',10)
        self.right_thrust_publisher = self.create_publisher(Float64, 'wamv/thrusters/right/thrust',10)
        self.left_thruster_pos_publisher = self.create_publisher(Float64, 'wamv/thrusters/left/pos',10)
        self.right_thruster_pos_publisher = self.create_publisher(Float64, 'wamv/thrusters/right/pos',10)

        self.psi = 0
        self.psi_error = 0
        self.prev_psi_d = 0
        self.psi_d = 0
        self.r = 0
        self.l_d = l_d
        self.r_tol = r_tol
        self.x_tol = x_tol
        self.wp_idx = 0
        self.wps = []
        self.X = 275 # forward thrust in N
        self.dt = 0.04
        self.err_int = 0
        
        # PID gains, based on pole placement
        self.m = 333
        self.T = 1
        self.k = self.T/self.m
        self.d = 1/self.k
        self.wn = 2
        self.zeta = 0.8
        self.Kp = self.m*self.wn**2-self.k
        self.Kd = 2*self.m*self.zeta*self.wn-self.d
        self.Ki = (self.wn/10)*self.Kp

        self.trigger = False
        self.wp_trigger = False
        self.timer_= self.create_timer(self.dt,self.los_pid_thrust_command)
        self.get_logger().info('LOS_PD_Controller publishing: ')

        self.psi_log = []
        self.psi_d_log = []
        self.ce_log = []

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
        
        if np.linalg.norm(self.cur_pos- self.wps[self.wp_idx+1]) < self.r_tol or np.linalg.norm(along_track) < self.x_tol:
                
            if self.wp_idx < len(self.wps)-2:
                self.wp_idx += 1
            else:
                self.wp_idx = len(self.wps)-1
                msg=Float64()
                msg.data=0.0
                self.left_thruster_pos_publisher.publish(msg)
                self.right_thruster_pos_publisher.publish(msg)
                return
               
        # adaptive lookahead dist
        # l_d = 8 * np.exp(-0.4 * np.linalg.norm(cross_track)) + 2        
        
        l_d = self.l_d
        
        los_point = proj + (l_d * path_vec)

        psi_d = math.atan2(los_point[1]-self.cur_pos[1], los_point[0]-self.cur_pos[0])
        self.psi_d = ssa(psi_d)
        
        self.r_d = ssa(self.psi_d - self.prev_psi_d)/self.dt
        self.prev_psi_d = self.psi_d
        
        # self.get_logger().info(f'Current position: {self.cur_pos}, Goal position: {self.wps[self.wp_idx+1]}, Distance: {np.linalg.norm(self.cur_pos - self.wps[self.wp_idx+1])}')
        # self.get_logger().info(f'CE: {np.linalg.norm(cross_track)}, LE: {np.linalg.norm(along_track)}')
        self.psi_error = ssa(self.psi_d - self.psi)
        
        # if not hasattr(self, 'err_int'):
        #     self.err_int = 0
        # self.err_int += self.psi_error*self.dt
        
        
        Y = 0
        # N = -self.Kp*self.psi_error - self.Kd*(self.r_d - self.r) - self.Ki*self.err_int
        # N = -self.Kp*self.psi_error - self.Kd*(self.r_d - self.r)
        N = -self.Kp*self.psi_error - self.Kd*self.r
        # X = self.X
        
        #thrust allocation:
        lp_x = -2.373
        ls_x = -2.373
        lp_y = -1.027
        ls_y = 1.027
        b = ls_y - lp_y
                
        t_p = self.X*0.5 + N/b
        t_s = self.X*0.5 - N/b
        
        self.publish_thrust_cmds([t_p,t_s,0.0,0.0])
       
        # self.psi_log.append(self.psi)
        # self.psi_d_log.append(psi_d)
        # self.ce_log.append(np.linalg.norm(cross_track))
    
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