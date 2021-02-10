#include "flightlib/envs/racing_env/racing_env.hpp"

namespace flightlib {

RacingEnv::RacingEnv() : RacingEnv(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/quadrotor_env.yaml")) {}

RacingEnv::RacingEnv(const std::string &cfg_path, const bool wave_track) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  loadParam(cfg_);

  Vector<3> scale_vector(0.5, 0.5, 0.5);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  quadrotor_ptr_->setSize(scale_vector);
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // update state
  quad_state_.x = Vector<25>::Zero();
  quad_state_.t = (Scalar) 0.0f;

  // define a bounding box
  world_box_ << -100, 100, -100, 100, -100, 100;
  quadrotor_ptr_->setWorldBox(world_box_);

  // airsim
  float uptilt_angle = 30.0;
  uptilt_angle = -(uptilt_angle / 90.0) * M_PI_2;
  Vector<3> B_r_BC(0.2, 0.0, 0.1);
  Matrix<3, 3> R_BC = Quaternion(std::cos(0.5 * uptilt_angle), 0.0, std::sin(0.5 * uptilt_angle), 0.0).toRotationMatrix();
  Matrix<3, 3> temp = Quaternion(std::cos(-0.5 * M_PI_2), 0.0, 0.0, std::sin(-0.5 * M_PI_2)).toRotationMatrix();
  R_BC = R_BC * temp;

  rgb_camera_ = std::make_unique<RGBCamera>();
  rgb_camera_->setFOV(racingenv::fov);
  rgb_camera_->setHeight(racingenv::image_height);
  rgb_camera_->setWidth(racingenv::image_width);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  image_height_ = racingenv::image_height;
  image_width_ = racingenv::image_width;

  // add gates, hard-coded for now
  for (int i = 0; i < racingenv::num_gates; i++) {
    gates_[i] = std::make_shared<StaticGate>("test_gate_" + std::to_string(i), "rpg_gate");
    gates_[i]->setPosition(Eigen::Vector3f(POSITIONS[i][0], POSITIONS[i][1], POSITIONS[i][2]));
    gates_[i]->setRotation(Quaternion(std::cos(ORIENTATIONS[i]), 0.0, 0.0, std::sin(ORIENTATIONS[i])));
  }
  setWaveTrack(wave_track);

  // add unity
  setUnity(true);
}

RacingEnv::~RacingEnv() {}

/*******************************
 * MAIN METHODS (STEP AND GET) *
 *******************************/

bool RacingEnv::step(const Ref<Vector<>> action) {
  // update command
  cmd_.t += sim_dt_;
  cmd_.collective_thrust = action[0];
  cmd_.omega = action.segment<3>(1);

  // simulate quadrotor
  bool success = quadrotor_ptr_->run(cmd_, sim_dt_);

  // update state
  quadrotor_ptr_->getState(&quad_state_);

  return success;
}

bool RacingEnv::getImage(Ref<ImageFlat<>> image) {
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();

    bool rgb_success = rgb_camera_->getRGBImage(cv_image_);

    if (rgb_success) {
      cv::split(cv_image_, cv_channels_);
      for (int i = 0; i < cv_image_.channels(); i++) {
        cv::cv2eigen(cv_channels_[i], channels_[i]);
        Map<ImageFlat<>> image_(channels_[i].data(), channels_[i].size());
        image.block<racingenv::image_height * racingenv::image_width, 1>(i * racingenv::image_height * racingenv::image_width, 0) = image_;
      }
    }
  } else {
    std::cout << "WARNING: Unity rendering not available; cannot get images." << std::endl;
    return false;
  }

  return true;
}

void RacingEnv::getState(Ref<Vector<>> state) {
  state.segment<QuadState::IDX::SIZE>(0) = quad_state_.x;
}

/****************************
 * METHODS RELATED TO UNITY *
 ****************************/

void RacingEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
  for (int i = 0; i < racingenv::num_gates; i++) {
    bridge->addStaticObject(gates_[i]);
  }
}

bool RacingEnv::setUnity(bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    // add this environment to Unity
    this->addObjectsToUnity(unity_bridge_ptr_);
  }
  return true;
}

bool RacingEnv::connectUnity(const int pub_port, const int sub_port) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_, pub_port, sub_port);
  return unity_ready_;
}

void RacingEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    std::cout << "WARNING: Flightmare Unity Bridge is not initialized." << std::endl;
  }
}

/************************
 * OTHER SETTER METHODS *
 ************************/

void RacingEnv::setReducedState(Ref<Vector<>> new_state) {
  quad_state_.x.segment<10>(0) = new_state;  // should maybe express this as sum instead of fixed number
  quadrotor_ptr_->setState(quad_state_);
}

void RacingEnv::setWaveTrack(bool wave_track) {
  float pos_z;
  int i;
  for (int j = 0; j < racingenv::num_elevated_gates; j++) {
    i = ELEVATED_GATES_INDICES[j];
    pos_z = POSITIONS[i][2];
    if (wave_track) {
      pos_z += 3.0;
    }
    // std::cout << pos_z << std::endl;
    gates_[i]->setPosition(Eigen::Vector3f(POSITIONS[i][0], POSITIONS[i][1], pos_z));
  }
}

bool RacingEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["test"]) {
    // load reinforcement learning related parameters
    test_yaml_ = cfg["test"]["test_yaml"].as<std::vector<std::vector<Scalar>>>();

    Eigen::MatrixXf test(3, 3);
    for (int i = 0; i < 3; i++) {
      test.row(i) = Eigen::VectorXf::Map(&test_yaml_[i][0], test_yaml_[i].size());
    }

    // std::cout << "YAML test:" << std::endl << test << std::endl;
  } else {
    return false;
  }

  return true;
}

}  // namespace flightlib