#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <string>

// yaml cpp
#include <yaml-cpp/yaml.h>

// opencv
#include <opencv2/core/mat.hpp>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include "flightlib/sensors/rgb_camera.hpp"

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
};
};
class RacingTestEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RacingTestEnv();
  RacingTestEnv(const std::string &cfg_path);
  ~RacingTestEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;  // TODO: add image as well

  // - public set functions
  bool loadParam(const YAML::Node &cfg);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliary functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);
  bool setUnity(bool render);
  bool connectUnity();
  void disconnectUnity();

  friend std::ostream &operator<<(std::ostream &os, const RacingTestEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"RacingTestEnv"};

  // observations and actions (for RL)
  Vector<racingenv::kNObs> quad_obs_;
  Vector<racingenv::kNAct> quad_act_;

  // image observations
  int image_counter_;
  cv::Mat image_;
  std::shared_ptr<RGBCamera> rgb_camera_;

  // unity
  std::shared_ptr<UnityBridge> unity_bridge_ptr_;
  SceneID scene_id_{UnityScene::WAREHOUSE};
  bool unity_ready_{false};
  bool unity_render_{false};

  // action and observation normalization (for learning)
  Vector<racingenv::kNAct> act_mean_;
  Vector<racingenv::kNAct> act_std_;
  Vector<racingenv::kNObs> obs_mean_ = Vector<racingenv::kNObs>::Zero();
  Vector<racingenv::kNObs> obs_std_ = Vector<racingenv::kNObs>::Ones();

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib