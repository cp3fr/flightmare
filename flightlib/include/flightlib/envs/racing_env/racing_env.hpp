#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>

// yaml cpp
#include <yaml-cpp/yaml.h>

// Eigen
#include <eigen3/Eigen/Dense>

// opencv
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>

// flightlib
#include "flightlib/envs/env_base_camera.hpp"
#include "flightlib/objects/static_gate.hpp"

namespace flightlib {

namespace racingenv {
    enum Ctl : int {
      // observations
      kObs = 0,
      //
      kPos = 0,
      kNPos = 3,
      kOri = 3,
      kNOri = 3,
      kLinVel = 6,
      kNLinVel = 3,
      kAngVel = 9,
      kNAngVel = 3,
      kNObs = 12,
      // control actions
      kAct = 0,
      kNAct = 4,
      // image dimensions
      // image_height = 600,
      // image_width = 800,
      // for testing
      image_height = 600,
      image_width = 800,
      fov = 80,
      // track info (should maybe be loaded)
      num_gates = 10,
      num_elevated_gates = 6,
    };
};

class RacingEnv final : public EnvBaseCamera {
 public:
  RacingEnv();
  RacingEnv(const std::string &cfg_path, const bool wave_track = false);
  ~RacingEnv();

  // method to set the quadrotor state and get a rendered image
  bool step(const Ref<Vector<>> action) override;
  bool getImage(Ref<ImageFlat<>> image) override;
  void getState(Ref<Vector<>> state) override;

  // Unity methods
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) override;
  bool setUnity(bool render) override;
  bool connectUnity(const int pub_port = 10253, const int sub_port = 10254) override;
  void disconnectUnity() override;

  // setter methods
  void setReducedState(const Ref<Vector<>> new_state);
  void setWaveTrack(bool wave_track);

  bool loadParam(const YAML::Node &cfg);

 private:
  // image observations (better way of doing this?)
  ImageChannel<racingenv::image_height, racingenv::image_width> channels_[3];

  // gates
  std::shared_ptr<StaticGate> gates_[racingenv::num_gates];

  std::vector<std::vector<Scalar>> test_yaml_;

  // constants?
  float POSITIONS[racingenv::num_gates][3] = {
    {-18.0,  10.0, 2.1},
    {-25.0,   0.0, 2.1},
    {-18.0, -10.0, 2.1},
    { -1.3,  -1.3, 2.1},
    {  1.3,   1.3, 2.1},
    { 18.0,  10.0, 2.1},
    { 25.0,   0.0, 2.1},
    { 18.0, -10.0, 2.1},
    {  1.3,  -1.3, 2.1},
    { -1.3,   1.3, 2.1},
  };
  float ORIENTATIONS[racingenv::num_gates] = {
    0.75 * M_PI_2,
    1.00 * M_PI_2,
    0.25 * M_PI_2,
    -0.25 * M_PI_2,
    -0.25 * M_PI_2,
    0.25 * M_PI_2,
    1.00 * M_PI_2,
    0.75 * M_PI_2,
    -0.75 * M_PI_2,
    -0.75 * M_PI_2,
  };
  int ELEVATED_GATES_INDICES[racingenv::num_elevated_gates] = {1, 3, 4, 6, 8, 9};
};

}  // namespace flightlib