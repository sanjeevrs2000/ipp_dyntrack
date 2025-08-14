#!/usr/bin/env python3
from rclpy.node import Node
import rclpy
import rclpy.time
from ultralytics import YOLO
from std_msgs.msg import Header
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from ros_gz_interfaces.msg import ParamVec
from rcl_interfaces.msg import Parameter
from cv_bridge import CvBridge
from dyntrack_planner.utils import FRONT_LEFT_K, BASE_LINK_TO_FRONT_LEFT_CAMERA_LINK_TF, SyncSubscription, pose_to_numpy
from dyntrack_planner.params import dir_path
import cv2
import numpy as np
import os

class Perception(Node):
    def __init__(self, model_path):
        super().__init__('perception_node')
        
        self.create_subscription(Odometry, 'wamv/sensors/position/ground_truth_odometry', self.sensor_callback, 10)


        self.bridge = CvBridge()

        camera_df = {'wamv/sensors/cameras/front_left_camera_sensor/image_raw': Image,
                    'wamv/sensors/cameras/front_right_camera_sensor/image_raw': Image,
                     }

        self.camera_sub = SyncSubscription(self, camera_df, self.img_callback, queue_size=10, approx_time=0.1)

        self.model = YOLO(model_path, verbose=False)
        self.pose = Pose()
        self.image_msg = Image()
        self.results_pub = self.create_publisher(ParamVec, 'wamv/yolo/results', 10)

        self.percept_timer = self.create_timer(timer_period_sec=0.2, callback=self.compute)

        self.base_cam_trans, self.base_cam_rot = BASE_LINK_TO_FRONT_LEFT_CAMERA_LINK_TF    
        self.baseline = 0.2
        self.K = FRONT_LEFT_K
        
        self.prev_pose = Pose()
        self.triggered_sensor = False
        self.triggered_image = False
        
        # self.image_pub = self.create_publisher(Image, '/depth_detection/visualization', 10)
        
    def img_callback(self, img1: Image, img2: Image):
        self.left_image = img1
        self.right_image = img2
        self.triggered_image = True

    def sensor_callback(self, pose: Odometry):
        
        self.pose = pose.pose.pose
        self.triggered_sensor = True

    def compute(self):
        if not self.triggered_sensor:
            return

        if not self.triggered_image:
            return
        
        self.triggered_sensor = False
        self.triggered_image = False

        if self.prev_pose is None:
            self.prev_pose = self.pose

        image = self.bridge.imgmsg_to_cv2(self.left_image, 'passthrough')
        left_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.bridge.imgmsg_to_cv2(self.right_image, 'passthrough')
        right_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        left_results = self.model(left_img)[0]
        right_results = self.model(right_img)[0]

        # Match detections between left and right images
        matched_pairs = self.match_detections(left_results, right_results)
        # Calculate 3D positions for matched pairs
        params = ParamVec()
        for left_box, right_box in matched_pairs:
            position = self.calculate_3d_position(left_box, right_box)
            if position is None:
                continue
            # Create detection message
            detection = self.create_detection_msg(left_box, position)
            params.params.append(detection)
        
        params.header.stamp = self.get_clock().now().to_msg()
        self.results_pub.publish(params)

    def match_detections(self, left_results, right_results):
        # Simple matching based on class and horizontal position
        matched = []
        used_right = set()
        for left_box in left_results.boxes:
            best_match = None
            best_diff = float('inf')
            left_y = left_box.xywh[0][1]
            # Iterate over each right detection
            for i, right_box in enumerate(right_results.boxes):
                # Only consider not-yet-matched right detection, and same class
                if i in used_right or left_box.cls != right_box.cls:
                    continue

                right_y = right_box.xywh[0][1]
                diff = abs(left_y - right_y)
                # Select the right detection with the minimum vertical difference within threshold
                if diff < 20 and diff < best_diff:
                    best_diff = diff
                    best_match = (i, right_box)

            if best_match is not None:
                used_right.add(best_match[0])
                matched.append((left_box, best_match[1]))

        return matched
    
    def calculate_3d_position(self, left_box, right_box):
        # Get centers of bounding boxes
        left_x, left_y = left_box.xywh[0][0].item(), left_box.xywh[0][1].item()
        right_x, right_y = right_box.xywh[0][0].item(), right_box.xywh[0][1].item()
        
        # Calculate disparity
        disparity = left_x - right_x
        
        # Focal length and principal point from intrinsic matrix
        f = self.K[0, 0]  # Assuming fx = fy
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        # Calculate 3D coordinates
        Z = (self.baseline * f) / (disparity + 1e-6)
        X = (left_x - cx) * Z / f
        Y = (left_y - cy) * Z / f
        
        cam_pos = np.array([Z,-X,-Y])
        if Z < 0:
            return None
        bl_pos = (BASE_LINK_TO_FRONT_LEFT_CAMERA_LINK_TF[1] @ cam_pos.T).T + BASE_LINK_TO_FRONT_LEFT_CAMERA_LINK_TF[0]
        trans, rot = pose_to_numpy(self.pose)
        w_pos = (rot @ bl_pos.T).T + trans
        return w_pos[:]

    def create_detection_msg(self, box, position):

        param = Parameter()
        
        l = int(box.cls.cpu().numpy())
        # param.name = box.names[l]
        param.value.double_value = box.conf.item()
        param.value.integer_value = l
        # print(box.xyxy.cpu().numpy()[0])
        param.value.integer_array_value = np.int32(box.xyxy.cpu().numpy()[0]).tolist()
        param.value.double_array_value = position.tolist()

        return param
    
def main(args=None):
    rclpy.init(args=args)
    path = os.path.join(dir_path, 'models/yolov8n_old.pt')
    node = Perception(path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
