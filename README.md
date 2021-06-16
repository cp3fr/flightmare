# Flightmare (customized version)

For general information on the Flightmare simulator, see the [parent repository](https://github.com/uzh-rpg/flightmare) of this one, as well as the [documentation](https://flightmare.readthedocs.io). This repository contains some changes and additional code used for my Master's thesis and a following project on optical flow.

The main changes to the code from the main repository as of June 16, 2021 are as follows:
- Physics and rendering are completely decoupled, i.e. one can make use of the rendering capabilities of Flightmare without having to use its implemented quadrotor dynamics
- Generation of ground-truth optical flow works in conjunction with the likewise customized [Flightmare Unity rendering engine](https://github.com/swengeler/flightmare_unity)
- Additional code adapts the [Deep Drone Acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) framework to work with Flightmare (and a simple MPC in Python, which means that the ROS software stack does not have to be used)

## Installation

### OpenCV requirements

## 'Racing' environment and its interface

## Optical flow
