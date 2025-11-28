import numpy as np
import scipy
from dyntrack_planner.params import d_max, theta, a0, d0, a1, d1
from dyntrack_planner.params import origin, grid_size, resolution
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from math import sin, cos, atan2, sqrt, pi, hypot, acos
import message_filters
import torch

### Mapping utils

# Inverse sensor models
def sensor_model(dist):
    
    # return the inverse observation model P(X|Z) as a function of the distance  
    if dist>35:
        px_z = 0.5
        
    else:
        px_z = (1/(1+ np.exp(a0*(dist - d0))))
    return px_z

def negative_sensor_model(dist):

    p = 1 - (1/(1+ np.exp(a1*(dist - d1))))
    return p

def world_to_grid(x, y):
    """ Converts world coordinates to grid indices """
    
    cell_x = int((x - origin[0]) / resolution)
    cell_y = int((y - origin[1]) / resolution)
    if 0 <= cell_x < grid_size[0] and 0 <= cell_y < grid_size[1]:
        return cell_x, cell_y
    
    return None

def grid_to_world(i, j):
    """ Converts grid indices to world coordinates """

    x = origin[0] + (i + 0.5) * resolution
    y = origin[1] + (j + 0.5) * resolution
    return x, y

#Current fov of vehicle
def get_fov(x, y, psi):
    
    # self.fov_count = int(2*theta*d*d/self.resolution**2)
    
    visible_ids = []       

    for i in range(int((x - origin[0] - d_max)//resolution),int((x - origin[0] + d_max)//resolution)):
        for j in range(int((y - origin[1] - d_max)//resolution), int((y - origin[1] + d_max)//resolution)):
            if i<0 or i>=grid_size[0] or j<0 or j>=grid_size[1]:
                continue

            cx = (i + 0.5) * resolution + origin[0]
            cy = (j + 0.5)* resolution + origin[1]
            dist = np.sqrt((cx-x)**2 + (cy-y)**2)
            
            phi = np.arctan2(cy-y, cx-x)
            delta_phi = (phi-psi + np.pi) % (2*np.pi) - np.pi 

            if dist <= d_max and np.abs(delta_phi) < theta:
                visible_ids.append([j,i])           

    return np.array(visible_ids)


### Sensor models and fov for batch inputs with torch

def batch_get_fov(x, y, psi, device):
    
    K = x.shape[0]

    x_coords = ((torch.arange(grid_size[0]) + 0.5) * resolution + origin[0]).to(device)
    y_coords = ((torch.arange(grid_size[1]) + 0.5) * resolution + origin[1]).to(device)

    Cx, Cy = torch.meshgrid(x_coords, y_coords)  # Note the order to get Cx, Cy

    dx = Cx.unsqueeze(0) - x.view(K, 1, 1)
    dy = Cy.unsqueeze(0) - y.view(K, 1, 1)
    dist = torch.sqrt(dx**2 + dy**2)
    phi = torch.atan2(dy, dx)
    delta_phi = (phi - psi.view(K, 1, 1) + np.pi) % (2 * np.pi) - np.pi
    
    fov_mask = (dist <= d_max) & (torch.abs(delta_phi) < theta)
    
    out = fov_mask.permute(0,2,1)

    return out

def batch_sensor_model(distances):
    # Handle distances > d_max
    result = torch.where(distances > d_max, 
                        torch.full_like(distances, 0.5),
                        1.0 / (1.0 + torch.exp(a0 * (distances - d0))))
    
    # Zero out where distances are 0 (non-FOV areas)
    result = torch.where(distances == 0, torch.zeros_like(result), result)
    
    return result

def batch_negative_sensor_model(distances):
    
    result = 1.0 - (1.0 / (1.0 + torch.exp(a1 * (distances - d1))))
    
    # Zero out where distances are 0 (non-FOV areas)
    result = torch.where(distances == 0, torch.zeros_like(result), result)
    
    return result


#### Entropy and information gain
def compute_entropy(p):

    if p == -1/100:
            return 1
    
    # Handle cases where log2 would be undefined
    if p <= 0 or p >= 1:
        return 0
    
    # Compute Bernoulli entropy for valid probabilities
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def compute_information_gain(p, d):

    return np.exp(-2*p) if p > 0.01 and p <= 1 else 0
    # return np.exp(-2*p - 0.001*d*d) if p > 0.01 and p < 1 else 0

def to_prob(l):
    
    # if l==0:
    #     return -1/100
    
    return 1 - (1/(1+np.exp(l)))

def to_logodds(p):
    
    if p == -1/100:
        return 0
    
    return np.log(p/(1-p))


#### message sync

class SyncSubscription:
    def __init__(self, node: Node, df, callback, queue_size=10, approx_time=0) -> None:
        if approx_time == 0:
            node.get_logger().debug(
                f'Subs to topics in sync : {",".join(df.keys())}')
            self.msgs_sub = message_filters.TimeSynchronizer(
                [message_filters.Subscriber(node, df[topic], topic) for topic in df], queue_size=queue_size)
        else:
            self.msgs_sub = message_filters.ApproximateTimeSynchronizer(
                [message_filters.Subscriber(node, df[topic], topic) for topic in df], 
                slop=approx_time, queue_size=queue_size, allow_headerless=True)
        self.msgs_sub.registerCallback(callback)



#### USV frame
# Partial credit to Marine Autonomous Vehicles Lab, IIT Madras, https://github.com/Akash7736/vrx23

FRONT_LEFT_K = np.array(
    [762.7223205566406, 0.0, 640.0, 0.0, 762.7223110198975, 360.0, 0.0, 0.0, 1.0]).reshape((3, 3))

BASE_LINK_TO_FRONT_LEFT_CAMERA_LINK_TF = (np.array([0.75, 0.1, 1.5]), np.array(
    [[0.9659258262890682, 0.0, 0.2588190451025209], [0.0, 1.0, 0.0], [-0.2588190451025209, 0.0, 0.9659258262890682]]))

BASE_LINK_TO_FRONT_LEFT_CAMERA_LINK_OPTICAL_TF = (np.array([0.75, 0.1, 1.5]), np.array(
    [[2.220446049250313e-16, -0.2588190451025208, 0.9659258262890682], [-1.0, 0.0, 1.6653345369377348e-16], [5.551115123125783e-17, -0.9659258262890682, -0.25881904510252074]]))

FAR_LEFT_CAMERA_LINK_TO_BASE_LINK_TF = (np.array([-0.3362158020630198, -0.3, -1.6430030232604929]), np.array(
    [[0.9659258262890682, -0.0, -0.2588190451025209], [0.0, 1.0, -0.0], [0.2588190451025209, 0.0, 0.9659258262890682]]))


FRONT_LEFT_CAMERA_LINK_TO_BASE_LINK_TF = (np.array([-0.3362158020630198, -0.1, -1.6430030232604929]), np.array(
    [[0.9659258262890682, -0.0, -0.2588190451025209], [0.0, 1.0, -0.0], [0.2588190451025209, 0.0, 0.9659258262890682]]))


FRONT_RIGHT_CAMERA_LINK_TO_BASE_LINK_TF = (np.array([-0.3362158020630198, 0.1, -1.6430030232604929]), np.array(
    [[0.9659258262890682, -0.0, -0.2588190451025209], [0.0, 1.0, -0.0], [0.2588190451025209, 0.0, 0.9659258262890682]]))


def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w

    rotation_matrix = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y -
                2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x -
                2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 *
                x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    return rotation_matrix


def pose_to_numpy(pose):
    rot_mat = quaternion_to_rotation_matrix(pose.orientation)
    trans = np.array([pose.position.x,
                      pose.position.y, pose.position.z])
    return trans, rot_mat

#### Bezier path
# Credit to PythonRobotics toolbox by Atsushi Sakai https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/BezierPath/bezier_path.py

def calc_4points_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset, n_points=100):
    """
    Compute control points and path given start and end position.

    :param sx: (float) x-coordinate of the starting point
    :param sy: (float) y-coordinate of the starting point
    :param syaw: (float) yaw angle at start
    :param ex: (float) x-coordinate of the ending point
    :param ey: (float) y-coordinate of the ending point
    :param eyaw: (float) yaw angle at the end
    :param offset: (float)
    :return: (numpy array, numpy array)
    """
    dist = np.hypot(sx - ex, sy - ey) / offset
    control_points = np.array(
        [[sx, sy],
         [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
         [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
         [ex, ey]])

    path = calc_bezier_path(control_points, n_points=n_points)

    return path, control_points


def calc_bezier_path(control_points, n_points=100):
    """
    Compute bezier path (trajectory) given control points.

    :param control_points: (numpy array)
    :param n_points: (int) number of points in the trajectory
    :return: (numpy array)
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def bernstein_poly(n, i, t):
    """
    Bernstein polynom.

    :param n: (int) polynom degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier(t, control_points):
    """
    Return one point on the bezier curve.

    :param t: (float) number in [0, 1]
    :param control_points: (numpy array)
    :return: (numpy array) Coordinates of the point
    """
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)


def bezier_derivatives_control_points(control_points, n_derivatives):
    """
    Compute control points of the successive derivatives of a given bezier curve.

    A derivative of a bezier curve is a bezier curve.
    See https://pomax.github.io/bezierinfo/#derivatives
    for detailed explanations

    :param control_points: (numpy array)
    :param n_derivatives: (int)
    e.g., n_derivatives=2 -> compute control points for first and second derivatives
    :return: ([numpy array])
    """
    w = {0: control_points}
    for i in range(n_derivatives):
        n = len(w[i])
        w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j])
                             for j in range(n - 1)])
    return w


def curvature(dx, dy, ddx, ddy):
    """
    Compute curvature at one point given first and second derivatives.

    :param dx: (float) First derivative along x axis
    :param dy: (float)
    :param ddx: (float) Second derivative along x axis
    :param ddy: (float)
    :return: (float)
    """
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)


class BoatVisualization:
    def __init__(self):
        # Define catamaran shape in local coordinates (boat frame)
        # Catamaran with two hulls connected by a deck - 1m length, thicker hulls, wider separation
        
        # Left hull coordinates (thicker hulls, 1m total length)
        x_left_hull = np.array([-0.5, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
        y_left_hull = np.array([0.25, 0.35, 0.42, 0.45, 0.45, 0.42, 0.35, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25])
        
        # Right hull coordinates (mirror of left hull)
        x_right_hull = x_left_hull.copy()
        y_right_hull = -y_left_hull.copy()
        
        # Connect the hulls with deck/bridge lines (wider separation)
        deck_x = np.array([0.5, 0.5, np.nan, -0.5, -0.5, np.nan, 0.0, 0.0])  # nan creates line breaks
        deck_y = np.array([0.25, -0.25, np.nan, 0.25, -0.25, np.nan, 0.25, -0.25])
        
        # Combine all coordinates
        self.x_local = np.concatenate([x_left_hull, [np.nan], x_right_hull, [np.nan], deck_x])
        self.y_local = np.concatenate([y_left_hull, [np.nan], y_right_hull, [np.nan], deck_y])
        
        # Scale factor: since coordinates are now in range [-0.5, 0.5], scale by 1.0 gives 1m length
        self.scale = 1.0  # 1 meter total length
        self.x_local *= self.scale
        self.y_local *= self.scale
    
    def transform_boat(self, x, y, psi):
        """
        Transform boat coordinates from local frame to global frame
        
        Args:
            x, y: position of boat center in global coordinates
            psi: heading angle in radians (0 = pointing right/east)
        
        Returns:
            x_global, y_global: transformed boat coordinates
        """
        # Rotation matrix
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        
        # Apply rotation
        x_rotated = cos_psi * self.x_local - sin_psi * self.y_local
        y_rotated = sin_psi * self.x_local + cos_psi * self.y_local
        
        # Apply translation
        x_global = x_rotated + x
        y_global = y_rotated + y
        
        return x_global, y_global
    
    def plot_boat(self, ax, x, y, psi, color='blue', alpha=0.7, linewidth=2, scale=None):
        """
        Plot the catamaran at given position and orientation
        
        Args:
            ax: matplotlib axis object
            x, y: position of boat center
            psi: heading angle in radians
            color: boat color
            alpha: transparency
            linewidth: line thickness
            scale: scale factor for boat size (default: self.scale)
        """
        # Use provided scale or default
        if scale is None:
            scale = self.scale
        # Temporarily scale shape for plotting
        x_local = self.x_local * (scale / self.scale)
        y_local = self.y_local * (scale / self.scale)
        # Transform
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        x_rotated = cos_psi * x_local - sin_psi * y_local
        y_rotated = sin_psi * x_local + cos_psi * y_local
        x_global = x_rotated + x
        y_global = y_rotated + y
        
        # Plot catamaran outline (handles NaN values for line breaks)
        ax.plot(x_global, y_global, color=color, linewidth=linewidth, alpha=alpha)
        
        # Fill the hulls separately (avoiding NaN issues)
        # Split coordinates at NaN values to fill hulls individually
        segments = []
        current_x, current_y = [], []
        
        for i in range(len(x_global)):
            if np.isnan(x_global[i]) or np.isnan(y_global[i]):
                if len(current_x) > 2:  # Only fill if we have enough points
                    segments.append((np.array(current_x), np.array(current_y)))
                current_x, current_y = [], []
            else:
                current_x.append(x_global[i])
                current_y.append(y_global[i])
        
        # Add the last segment if it exists
        if len(current_x) > 2:
            segments.append((np.array(current_x), np.array(current_y)))
        
        # Fill each hull
        for seg_x, seg_y in segments:
            if len(seg_x) > 2:  # Only fill closed shapes
                ax.fill(seg_x, seg_y, color=color, alpha=alpha*0.5)
        
        # Add a direction indicator (arrow pointing forward)
        arrow_length = scale * 0.3
        ax.arrow(x, y, arrow_length * np.cos(psi), arrow_length * np.sin(psi),
                head_width=scale*0.08, head_length=scale*0.12, 
                fc=color, ec=color, alpha=alpha)
        
        return x_global, y_global
    
    
### Dubins path - Atsushi Sakai github.com/AtsushiSakai/PythonRobotics

def dubins_path_npoints(sx, sy, syaw, ex, ey, eyaw, curvature, n_points = 25):
    
    x,y,_,_,_ = plan_dubins_path(sx, sy, syaw, ex, ey, eyaw, curvature, step_size=0.2)

    dx = np.diff(x)
    dy = np.diff(y)
    dl = np.hypot(dx, dy)
    
    L = np.concatenate(([0], np.cumsum(dl)))
    total_len = L[-1]

    target_distances = np.linspace(0, total_len, n_points)
    
    # interpolate points to find equally space points
    x_interp = np.interp(target_distances, L, x)
    y_interp = np.interp(target_distances, L, y)
    path = np.column_stack([x_interp, y_interp])

    return path


def plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature,
                     step_size=0.1, selected_types=None):
    """
    Plan dubins path

    Parameters
    ----------
    s_x : float
        x position of the start point [m]
    s_y : float
        y position of the start point [m]
    s_yaw : float
        yaw angle of the start point [rad]
    g_x : float
        x position of the goal point [m]
    g_y : float
        y position of the end point [m]
    g_yaw : float
        yaw angle of the end point [rad]
    curvature : float
        curvature for curve [1/m]
    step_size : float (optional)
        step size between two path points [m]. Default is 0.1
    selected_types : a list of string or None
        selected path planning types. If None, all types are used for
        path planning, and minimum path length result is returned.
        You can select used path plannings types by a string list.
        e.g.: ["RSL", "RSR"]

    Returns
    -------
    x_list: array
        x positions of the path
    y_list: array
        y positions of the path
    yaw_list: array
        yaw angles of the path
    modes: array
        mode list of the path
    lengths: array
        arrow_length list of the path segments.

    Examples
    --------
    You can generate a dubins path.

    >>> start_x = 1.0  # [m]
    >>> start_y = 1.0  # [m]
    >>> start_yaw = np.deg2rad(45.0)  # [rad]
    >>> end_x = -3.0  # [m]
    >>> end_y = -3.0  # [m]
    >>> end_yaw = np.deg2rad(-45.0)  # [rad]
    >>> curvature = 1.0
    >>> path_x, path_y, path_yaw, mode, _ = plan_dubins_path(
                start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)
    >>> plt.plot(path_x, path_y, label="final course " + "".join(mode))
    >>> plot_arrow(start_x, start_y, start_yaw)
    >>> plot_arrow(end_x, end_y, end_yaw)
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.axis("equal")
    >>> plt.show()

    .. image:: dubins_path.jpg
    """
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]

    # calculate local goal x, y, yaw
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, modes, lengths = _dubins_path_planning_from_origin(
        local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size,
        planning_funcs)

    # Convert a local coordinate path to the global coordinate
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

    return x_list, y_list, yaw_list, modes, lengths


def _mod2pi(theta):
    return angle_mod(theta, zero_2_2pi=True)


def _calc_trig_funcs(alpha, beta):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


_PATH_TYPE_MAP = {"LSL": _LSL, "RSR": _RSR, "LSR": _LSR, "RSL": _RSL,
                  "RLR": _RLR, "LRL": _LRL, }


def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                      step_size, planning_funcs):
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    for planner in planning_funcs:
        d1, d2, d3, mode = planner(alpha, beta, d)
        if d1 is None:
            continue

        cost = (abs(d1) + abs(d2) + abs(d3))
        if best_cost > cost:  # Select minimum length one.
            b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost

    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list = _generate_local_course(lengths, b_mode,
                                                      curvature, step_size)

    lengths = [length / curvature for length in lengths]

    return x_list, y_list, yaw_list, b_mode, lengths


def _interpolate(length, mode, max_curvature, origin_x, origin_y,
                 origin_yaw, path_x, path_y, path_yaw):
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:  # curve
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)

        if mode == "L":  # left turn
            path_yaw.append(origin_yaw + length)
        elif mode == "R":  # right turn
            path_yaw.append(origin_yaw - length)

    return path_x, path_y, path_yaw


def _generate_local_course(lengths, modes, max_curvature, step_size):
    p_x, p_y, p_yaw = [0.0], [0.0], [0.0]

    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue

        # set origin state
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]

        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature,
                                           origin_x, origin_y, origin_yaw,
                                           p_x, p_y, p_yaw)
            current_length += step_size

        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x,
                                       origin_y, origin_yaw, p_x, p_y, p_yaw)

    return p_x, p_y, p_yaw

def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle