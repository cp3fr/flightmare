#include "flightlib/envs/racing_env/racing_env.hpp"

namespace flightlib {

RacingEnv::RacingEnv()
  : RacingEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/racing_test_env.yaml")) {}

RacingEnv::RacingEnv(const std::string &cfg_path)
  : EnvBaseCamera() {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  // load parameters
  loadParam(cfg_);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);
  // scale quadrotor so that the camera view is unobstructed
  Vector<3> scale_vector(0.5, 0.5, 0.5);
  quadrotor_ptr_->setSize(scale_vector);

  // define a bounding box (large enough for the track)
  world_box_ << -30, 30, -30, 30, 0, 30;
  quadrotor_ptr_->setWorldBox(world_box_);

  // not entirely sure what this has to be used for...
  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<racingenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<racingenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // define input and output dimension for the environment
  obs_dim_ = racingenv::kNObs;
  act_dim_ = racingenv::kNAct;

  // camera positioning and orientation (same as AlphaPilot simulator)
  // camera positioning and orientation (same as AlphaPilot simulator)
  // positioning from the center of the quadrotor
  Vector<3> B_r_BC(0.2, 0.0, 0.1);
  // camera is rotated by 90° to the right and tilted up by 30°
  float uptilt_angle = -(30.0 / 90.0) * M_PI_2;
  Matrix<3, 3> R_BC = Quaternion(std::cos(0.5 * uptilt_angle), 0.0, std::sin(0.5 * uptilt_angle), 0.0).toRotationMatrix();
  Matrix<3, 3> temp = Quaternion(std::cos(-0.5 * M_PI_2), 0.0, 0.0, std::sin(-0.5 * M_PI_2)).toRotationMatrix();
  R_BC = R_BC * temp;

  // add camera
  rgb_camera_ = std::make_unique<RGBCamera>();
  rgb_camera_->setFOV(racingenv::fov);
  rgb_camera_->setHeight(racingenv::image_height);
  rgb_camera_->setWidth(racingenv::image_width);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  // add gates, hard-coded for now
  float positions[racingenv::num_gates][3] = {
    {-18.0,  10.0, 2.5},
    {-25.0,   0.0, 2.5},
    {-18.0, -10.0, 2.5},
    { -1.3,  -1.3, 2.5},
    {  1.3,   1.3, 2.5},
    { 18.0,  10.0, 2.5},
    { 25.0,   0.0, 2.5},
    { 18.0, -10.0, 2.5},
    {  1.3,  -1.3, 2.5},
    { -1.3,   1.3, 2.5},
  };
  // only need the rotation angle around the z-axis
  float orientations[racingenv::num_gates] = {
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
  // sizes also just estimated based on visual inspection
  for (int i = 0; i < racingenv::num_gates; i++) {
    gates_[i] = std::make_shared<StaticGate>("test_gate_" + std::to_string(i), "rpg_gate");
    gates_[i]->setPosition(Eigen::Vector3f(positions[i][0], positions[i][1], positions[i][2]));
    gates_[i]->setRotation(Quaternion(std::cos(orientations[i]), 0.0, 0.0, std::sin(orientations[i])));
    gates_[i]->setSize(Vector<3>(1.2, 1.0, 1.2));
  }

  // add unity
  setUnity(true);
}

RacingEnv::~RacingEnv() {}

/*
 * Step, reset and observation methods
 */

bool RacingEnv::getObs(Ref<Vector<>> state_obs, Ref<ImageFlat<>> image_obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle and set the state observation vector
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;
  state_obs.segment<racingenv::kNObs>(racingenv::kObs) = quad_obs_;

  // also capture an image and set the image observation vector
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();

    bool rgb_success = rgb_camera_->getRGBImage(cv_image_);

    if (rgb_success) {
      cv::split(cv_image_, cv_channels_);
      for (int i = 0; i < cv_image_.channels(); i++) {
        cv::cv2eigen(cv_channels_[i], channels_[i]);
        Map<ImageFlat<>> image_(channels_[i].data(), channels_[i].size());
        image_obs.block<racingenv::image_height * racingenv::image_width, 1>(i * racingenv::image_height * racingenv::image_width, 0) = image_;
      }
    }
  } else {
    logger_.warn("Unity rendering not available; cannot get images.");
  }

  return true;
}

Scalar RacingEnv::step(const Ref<Vector<>> act, Ref<Vector<>> state_obs, Ref<ImageFlat<>> image_obs) {
  // TODO: figure out what the mean and std should be...
  // => maybe the max thrust is so low because it's mass-normalised?
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(state_obs, image_obs);

  return 0.0;
}

bool RacingEnv::reset(Ref<Vector<>> state_obs, Ref<ImageFlat<>> image_obs, const bool random) {
  // no random initialisation for now (since it would depend on the trajectory)
  // => maybe instead, reset with a given state?

  quad_state_.setZero();
  quad_act_.setZero();
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // obtain observations
  getObs(state_obs, image_obs);

  return true;
}

/*
 * Setter methods
 */

bool RacingEnv::setReducedState(Ref<Vector<10>> reduced_state) {
  quad_state_.x.segment<10>(0) = reduced_state;
  quadrotor_ptr_->setState(quad_state_);

  return true;
}

/*
 * Getter methods
 */

int RacingEnv::getImageHeight() const {
  return racingenv::image_height;
}

int RacingEnv::getImageWidth() const {
  return racingenv::image_width;
}

/*
 * Unity-related methods
 */

void RacingEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
  //bridge->addStaticObject(gate_);
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
    logger_.info("Flightmare Bridge is created.");
  }
  return true;
}

bool RacingEnv::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

void RacingEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

/*
 * Parameters
 */

bool RacingEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["racing_env"]) {
    sim_dt_ = cfg["racing_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["racing_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  return true;
}

/*
 * Print formatting
 */

std::ostream &operator<<(std::ostream &os, const RacingEnv &quad_env) {
  os.precision(3);
  os << "Racing Test Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n";
  os.precision();
  return os;
}

}  // namespace flightlib