# An Informative Path Planning Framework for Target Tracking and Active Mapping in Dynamic Environments

This repository contains code for the RA-L submission "An Informative Planning Framework for Target Tracking and
Active Mapping in Dynamic Environments with ASVs".

## Setup the docker environment

The simulation experiments are performed with the [VRX simulator](https://github.com/osrf/vrx). The software has been developed and tested with the simulator in Ubuntu 22.04 with ROS2 Humble and Gazebo garden. To get started, build a docker image by following the setup instructions. First clone the repository:
```
git clone git@github.com:sanjeevrs2000/ipp_dyntrack.git
```

Build the docker image, and then open a container with the newly built image:
```
docker build -f Dockerfile -t user/ipp_dyntrack:1.0
chmod +x run_docker.sh
./run_docker.sh
```

## How to use the repository?

The packages ```dyntrack_planner``` contains nodes related to our IPP framework for perception, mapping and adaptive planning with the WAM-V ASV. The package ```asv_control``` implements a path following PID controller for the ASV based on line of sight (LOS) guidance. In the ```vrx_gz``` package, some files have been modified from the original simulator for the sake of our experiments.

### Run experiments with the simulator

To access a bash terminal inside the container, open a new terminal and run: 
```
docker exec -it ipp_dyntrack_container bash
``` 

To run an instance of the IPP framework inside the docker container, run the launch file:
```
ros2 launch dyntrack_planner informative_planner_log.launch.py headless:=True 
# headless:=False if you need to see the GUI
```

You can tune some parameters related to the experiments in ```src/dyntrack_planner/dyntrack_planner/params.py```. To visualize with rviz2, open another bash terminal inside the container and run:
```
ros2 launch dyntrack_planner vis_rviz.launch.py
```

### Spatiotemporal prediction network

To view the setup for dataset generation, training and testing of the spatiotemporal prediction network, go to the ```src/spatiotemp_pred_nn``` directory. To see sample predictions with the network, inside the container:
```
cd src/spatiotemporal_pred_nn/src
python3 test.py
```
To run a training instance, run ```python3 train.py```.