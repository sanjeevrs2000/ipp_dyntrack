#!/bin/bash
# Basic entrypoint for ROS / Colcon Docker containers

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash
echo "Sourced ROS 2 ${ROS_DISTRO}"

# Source the base workspace, if built
if [ -f /asv_ws/install/setup.bash ]
then
  source /vrx_ws/install/setup.bash
  echo "Sourced vrx_ws base workspace"
fi

# Execute the command passed into this entrypoint
exec "$@"
