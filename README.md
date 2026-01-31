# An Informative Path Planning Framework for Target Tracking and Active Mapping in Dynamic Environments

This repository contains code for the RA-L paper "An Informative Planning Framework for Target Tracking and
Active Mapping in Dynamic Environments with ASVs".

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FLRA.2026.3653335-blue)](https://doi.org/10.1109/LRA.2026.3653335)
[![arXiv](https://img.shields.io/badge/arXiv-2508.14636-B31B1B.svg)](https://arxiv.org/abs/2508.14636)

The paper can be found [here](https://arxiv.org/abs/2508.14636). If you found this repository useful, feel free to cite it.

```
@article{sanjeev2026informative,
  title={An informative planning framework for target tracking and active mapping in dynamic environments with asvs},
  author={Ramkumar Sudha, Sanjeev and Popovi{\'c}, Marija and Coates, Erlend M},
  journal={IEEE Robotics and Automation Letters},
  volume={11},
  number={3},
  pages={2690--2697},
  year={2026},
}
```

## Setup the docker environment

The simulation experiments are performed with the [VRX simulator](https://github.com/osrf/vrx). The software has been developed and tested with the simulator in Ubuntu 22.04 with ROS2 Humble and Gazebo garden. To get started, build a docker image by following the setup instructions. First clone the repository in your workspace:
```
git clone git@github.com:sanjeevrs2000/ipp_dyntrack.git
```

Then build the docker image:
```
docker build -f Dockerfile -t user/ipp_dyntrack:1.0 .
```

To open a container with the newly built image:
```
chmod +x run_docker.sh
./run_docker.sh
```
Note: To use GPU acceleration with NVIDIA GPUs, ensure that you have [nvidia-ctk](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed. If you do not have a GPU, in ```docker-compose.yaml``` comment the ```runtime: nvidia``` argument before running ```docker compose up``` as it might cause some errors.

## How to use the repository?

The packages ```dyntrack_planner``` contains nodes related to implementation of our IPP framework. The package ```asv_control``` implements a path following PD controller for the ASV based on line of sight (LOS) guidance. In the ```vrx_gz``` package, some files have been modified from the original simulator for the sake of our experiments.
<!-- The nodes for perception (Sec. IV B in the paper), mapping (Sec. IV C), and adaptive planning (Sec. IV E) are `perception.py`, `grid_map.py`, and `rec_horizon_planner.py`, respectively.  -->

### Run experiments with the simulator

To access a bash terminal inside the container, open a new terminal and run: 
```
docker exec -it ipp_dyntrack_container bash
``` 
First, build the packages and then source the workspace in the container:
```
colcon build --merge-install
source install/setup.bash
```
To run an instance of the IPP framework inside the docker container, run the launch file:
```
ros2 launch dyntrack_planner informative_planner.launch.py headless:=True 
# run without headless:=False to view the Gazebo GUI
```
To visualize with rviz2, open another bash terminal inside the container and run:
```
ros2 launch dyntrack_planner vis_rviz.launch.py
```

For further experimentation, you can tune some parameters related to the experiments in ```src/dyntrack_planner/dyntrack_planner/params.py```. 

<!-- Remember to colcon build and source the workspace before running the files after making changes to the src. -->

### Spatiotemporal prediction network

To view the setup for dataset generation, training and testing of the spatiotemporal prediction network, go to the ```src/spatiotemp_pred_nn``` directory. To run sample predictions with the network, inside the container:
```
cd src/spatiotemporal_pred_nn/src
python3 test.py
```
To run a training instance, run ```python3 train.py```.