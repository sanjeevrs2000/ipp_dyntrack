# Informative Path Planning Framework for Target Tracking and Active Mapping in Dynamic Environments

This repository contains code for an informative path planning (IPP) framework for active mapping of moving targets in dynamic environments with an ASV.

## Setup the simulation environment

This code has been developed and tested with the VRX simulator in Ubuntu 22.04.
To get started, install ROS2 Humble and Gazebo Garden.
Then install additional dependencies:
```

```

Create a workspace `path/to/your_ws` 
Then create a src directory ```mkdir src``` and ```cd /path/to/your_ws```.

Now clone the repository ```git clone ```

Install the required python libraries from `requirements.txt`

Source the ROS installation ```source /opt/ros/humble/setup.bash```. Now, build the ROS2 packages with colcon : ```colcon build --merge-install```

Now source the workspace ```source install/setup.bash/```


## How to use the repository?

### Run experiments with the simulator

To run an instance of our IPP framework with the adaptive planner by running the launch file:
```ros2 launch dyntrack_planner informative_planner_log_launch.py```

For visualization in Rviz, run ```ros2 launch ```

You can change the configurations and parameters of the approach in ```dyntrack_planner/dyntrack_planner/params.py```. Before starting, also set the path to your directory in ```params.py``` to point to the right folders for loading the models and saving logs`. 
Remember to build and source the repository to make sure that the changes are reflected.

### Spatiotemporal prediction network

To view the setup for dataset generation, training and testing of our spatiotemporal network, check the ```spatiotemp_pred_nn``` directory.