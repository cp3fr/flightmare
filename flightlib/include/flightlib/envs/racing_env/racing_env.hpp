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
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base_camera.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/objects/static_gate.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

namespace flightlib {

using ChannelMatrix = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ChannelStride = Eigen::Stride<Eigen::Dynamic, 3>;
template<typename S>
using ChannelMap = Eigen::Map<ChannelMatrix, Eigen::Unaligned, S>;

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
  image_height = 600,
  image_width = 800,
  fov = 120,
  // track info (should probably be loaded)
  num_gates = 10,
};
};
class RacingEnv final : public EnvBaseCamera {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RacingEnv();
  RacingEnv(const std::string &cfg_path);
  ~RacingEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> state_obs, Ref<ImageFlat<>> image_obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> state_obs, Ref<ImageFlat<>> image_obs) override;
  bool getObs(Ref<Vector<>> state_obs, Ref<ImageFlat<>> image_obs) override;

  // - public set functions
  bool setReducedState(Ref<Vector<10>> reduced_state);
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  int getImageHeight() const;
  int getImageWidth() const;

  // - auxiliary functions
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);
  bool setUnity(bool render);
  bool connectUnity();
  void disconnectUnity();

  friend std::ostream &operator<<(std::ostream &os, const RacingEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Matrix<3, 2> world_box_;

  // ?
  Vector<racingenv::kNAct> act_mean_;
  Vector<racingenv::kNAct> act_std_;

  // observations and actions (for RL)
  Vector<racingenv::kNObs> quad_obs_;
  Vector<racingenv::kNAct> quad_act_;

  // camera
  std::shared_ptr<RGBCamera> rgb_camera_;

  // image observations
  ImageChannel<racingenv::image_height, racingenv::image_width> channels_[3];
  cv::Mat cv_image_;
  cv::Mat cv_channels_[3];

  // gate(s)
  std::shared_ptr<StaticGate> gates_[10];

  // Unity
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  SceneID scene_id_{UnityScene::WAREHOUSE};
  bool unity_ready_{false};
  bool unity_render_{false};

  // IO
  YAML::Node cfg_;
  Logger logger_{"RacingEnv"};
};

}  // namespace flightlib