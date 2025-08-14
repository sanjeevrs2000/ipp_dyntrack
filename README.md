# Informative Path Planning Framework for Target Tracking and Active Mapping in Dynamic Environments

This repository contains code for our RA-L submission "An Informative Planning Framework for Target Tracking and
Active Mapping in Dynamic Environments with ASVs"

## Setup the docker environment

First clone the repository ```git clone ```

To install the VRX-simulator and other dependencies, first setup the docker environment and open an instance of the built image:
```
docker build -f Dockerfile -t user/ipp_dyntrack:1.0
docker compose up
```

In another terminal, run:```docker exec -it ipp_dyntrack_container bash``` to open a bash terminal inside the container.

## How to use the repository?

### Run experiments with the simulator

To run an instance of the IPP framework inside the docker container, run the launch file:
```
ros2 launch dyntrack_planner informative_planner_log_launch.py

```
To visualize in Rviz, run:
```
ros2 launch 
```

You can change the configurations and parameters for the mapping and planning in ```src/dyntrack_planner/dyntrack_planner/params.py```. 

### Spatiotemporal prediction network

To view the setup for dataset generation, training and testing of the spatiotemporal prediction network, check the ```spatiotemp_pred_nn``` directory.