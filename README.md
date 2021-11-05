# Flightmare (customized version)

For general information on the Flightmare simulator, see the [parent repository](https://github.com/uzh-rpg/flightmare) of this one, as well as the [documentation](https://flightmare.readthedocs.io). This repository contains some changes and additional code used for my Master's thesis and a following project on optical flow.

The main changes to the code from the main repository as of June 16, 2021 are as follows:
- Physics and rendering are completely decoupled, i.e. one can make use of the rendering capabilities of Flightmare without having to use its implemented quadrotor dynamics
- Generation of ground-truth optical flow works in conjunction with the likewise customized [Flightmare Unity rendering engine](https://github.com/swengeler/flightmare_unity)
- Additional code adapts the [Deep Drone Acrobatics](https://github.com/uzh-rpg/deep_drone_acrobatics) framework to work with Flightmare (and a simple MPC in Python, which means that the ROS software stack does not have to be used)

## Installation

The dependencies outlined in the original Flightmare documentation should be installed. Installing the Python dependencies is possible using the provided conda `environment.yaml` file. The part that can be a bit trickier is installing the `flightgym` package, which provides an interface in Python for using the Flightmare simulator.

### OpenCV

The main issue with compiling Flightmare that I have found is getting the right OpenCV version as well as a small change that is required in the actual OpenCV source code before it can successfully be used with some of the new code I added (particularly for dealing with receiving images/optical flow data from the Flightmare Unity application). In the following, I try to provide instructions that should *hopefully* work.

First, I have found everything to work best when OpenCV 4 is installed. There are several options to get the development files to compile Flightmare against. The easiest is probably to use the system package manager (which might not be the correct version). The second option is to install `libopencv` using `conda`. The third (and most tedious) option is to clone/download the [OpenCV source code](https://github.com/opencv/opencv).

The next step is to modify one of the files of the OpenCV source code, namely `$OPENCV_INCLUDE_PATH/core/eigen.hpp ` (e.g. `OPENCV_INCLUDE_PATH=/usr/include/opencv4/opencv2` when installed using `apt` on Ubuntu). One of the existing functions simply has to be copied with a slightly different function signature (see [here](https://github.com/opencv/opencv/issues/16606) for more information):

```cpp
// FUNCTION TO BE ADDED
template<typename _Tp>  static inline
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        const Mat _dst(src.cols, src.rows, traits::Type<_Tp>::value,
             dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
    }
    else
    {
        const Mat _dst(src.rows, src.cols, traits::Type<_Tp>::value,
                 dst.data(), (size_t)(dst.outerStride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
    }
}
```

The above is simplest to do with a system-wide installation of OpenCV, which requires admin rights (for installation and modification of the code). As mentioned above, installing OpenCV in a local `conda` environment (from the `conda-forge` channel, since no OpenCV 4 version is available in the default channel) is probably the next-best option and the same steps as above can be followed.

If this is also not possible for some reason, the [manual installation instructions](https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html) for OpenCV (for the desired version) can be followed. The source code should still be modified. One issue I found with this method is that unlike what the linked page says, the CMake files required for this project's CMake configuration to find everything are not actually located at `$INSTALL_DIR/cmake/opencv4`, but rather under `$INSTALL_DIR/lib/cmake`. Because of this, I included this path manually in the main `CMakeLists.txt` (namely the line `find_package(OpenCV 4 REQUIRED PATHS /home/simon/.local/lib/cmake)`).

For all of the above cases, **make sure that the output of running CMake (see below) shows that the right OpenCV location has been found**.

### Eigen

While Eigen did not create nearly as many problems as OpenCV, there are two issues that I encountered, that I managed to find a fix for. This is just some information in case similar issues occur despite the implemented fixes.

The first is related to `<unsupported/Eigen/CXX11/Tensor>` apparently not being found during compilation (although this did not always happen, and the files are present...). Since we do not need tensor support for using Flightmare, this can simply be disabled by adding `add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)` to the main `CMakeLists.txt` file (which has already been done for this project).

The second issue is related to receiving large image data from the Flightmare Unity application. This is done (more or less) by receiving it using OpenCV and then converting the data to Eigen. If the images are large (i.e. high image resolution), Eigen might cause crashes, since there is a limit set to how large a matrix can be allocated for this data (at least this is my understanding). The problem can be resolved by raising that limit with `add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=3145728)` (as an example value) added to the `CMakeLists.txt` (which is also already done for this project).

### Building the package

Once all of the above is taken care of, the basic steps are as follows. First go to the build directory:

```shell
cd flightlib/build
```

Run CMake to generate build files (the `CMakeLists.txt` file from the parent directory is used):

```shell
cmake ..
```

Compile everything:

```shell
make
```

This should generate (next to all the other files) `flightgym.cpython-39-x86_64-linux-gnu.so`. One aspect to pay attention to here is that the name of this file depends on the Python version that is found by CMake (i.e. the `39` stands for Python 3.9). 

To install the Python package, that file needs to be copied to the correct location and its name is hard-coded in `flightlib/build/setup.py`. Thus, depending on the Python version used to install Flightmare, the name of the file might have to be changed there. 

By default Python 3.9 is assumed to be used, which is **NOT** the one installed by the `environment.yaml` file, because this repository was used more recently by a different project using Python 3.9.

Once all of the above is figured out however, the `flightgym` package can be installed from the build directory with:

```shell
pip install .
```

## Running DDA

As mentioned above, this repository re-implements DDA to work with Flightmare. Here I will explain shortly how to train and test models. Testing in this case means actually flying (test) trajectories rather than predicting control commands offline on unseen data.

### Prerequisites

The training is performed by loading a trajectory (or multiple) from CSV file(s) and then flying that trajectory (these trajectories) repeatedly. Some trajectories from the [AlphaPilot dataset](https://osf.io/gvdse/wiki/home/) are included in `dda-inputs`, which will be stored on one of the RPG servers. Also included are checkpoints for the attention/gaze prediction models used by some of the implemented models that can be trained for DDA.

These trajectories should be specified as absolute paths in the specification YAML file. In contrast, the output data is stored relative to the environment variable `DDA_ROOT`. An additional environment variable that should be set is `FLIGHTMARE_PATH`, which should point to the root of this repository.

To run either the training or testing script, the Flightmare Unity application of course needs to be running. In addition, the correct ports for communication between the server and client need to be specified (which also allows us to run multiple instances in parallel). That means that the `pub_port` parameter in the specification YAML file (or adjusted manually for the testing script) and the `-input-port` of the Flightmare Unity application need to match. The same needs to be the case for the `sub_port` and `-output-port` parameters.

### Training

All options for training the models should be specified in a YAML file that is used as input for the training script. There is an example template (that trains on a single trajectory) under `flightil/dda/config/template.yaml`. More up-to-date examples for the main models compared for this project are also stored in the same location (`*totmedtraj.yaml`). These use multiple trajectories for training.

To actually run training, go to `flighil/dda` and run:

```shell
python learning_iterative.py --settings_file <path_to_settings_file>
```

### Testing

Once training is done, and the data has been saved under `DDA_ROOT`, the model can be tested by flying trajectories (train or test) by itself, or by letting an MPC fly the trajectory and recording the command predictions. The training script saves a few model checkpoints, the latest one of which should be provided as the model load path for the testing script.

An example use of the testing script would be to go to `flightil/dda` and run:

```shell
python test.py -mlp <path_to_last_model_checkpoint> -tp $DDA_INPUTS_PATH/mtt_total_median/flat/test
```

This runs the specified model on the test trajectories included in `dda-inputs` (should be backed up by RPG). For more options of the testing script, see the script itself (or `python test.py -h`). 
