#include "flightlib/envs/racing_env/racing_test_env.hpp"

namespace flightlib {

RacingTestEnv::RacingTestEnv()
  : RacingTestEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/racing_test_env.yaml")) {}

RacingTestEnv::RacingTestEnv(const std::string &cfg_path)
  : EnvBase() {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  world_box_ << -20, 20, -20, 20, 0, 20;
  quadrotor_ptr_->setWorldBox(world_box_);

  // define input and output dimension for the environment
  obs_dim_ = racingenv::kNObs;
  act_dim_ = racingenv::kNAct;

  // add camera
  image_counter_ = 0;
  rgb_camera_ = std::make_unique<RGBCamera>();
  Vector<3> B_r_BC(0.0, 0.0, -0.3);
  Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  rgb_camera_->setFOV(90);
  rgb_camera_->setWidth(720);
  rgb_camera_->setHeight(480);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  // add unity
  setUnity(true);

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<racingenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<racingenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // load parameters
  loadParam(cfg_);
}

RacingTestEnv::~RacingTestEnv() {}

bool RacingTestEnv::reset(Ref<Vector<>> obs, const bool random) {
  image_counter_ = 0;
  quad_state_.setZero();
  quad_act_.setZero();

  if (random) {
    // randomly reset the quadrotor state
    // reset position
    quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) + 5;
    if (quad_state_.x(QS::POSZ) < -0.0)
      quad_state_.x(QS::POSZ) = -quad_state_.x(QS::POSZ);
    // reset linear velocity
    quad_state_.x(QS::VELX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::VELZ) = uniform_dist_(random_gen_);
    // reset orientation
    quad_state_.x(QS::ATTW) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_);
    quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_);
    quad_state_.qx /= quad_state_.qx.norm();
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool RacingTestEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  // convert quaternion to euler angle
  Vector<3> euler_zyx = quad_state_.q().toRotationMatrix().eulerAngles(2, 1, 0);
  // quaternionToEuler(quad_state_.q(), euler);
  quad_obs_ << quad_state_.p, euler_zyx, quad_state_.v, quad_state_.w;

  // see here: https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
  // apparently this just means that we assign to a segment of kNObs entries, starting at position
  //  kObs of the vector obs (in this case this just seems to be the start of the vector)
  obs.segment<racingenv::kNObs>(racingenv::kObs) = quad_obs_;

  // also capture an image
  // for now just print some information, since I don't know how the conversion from C++ to numpy should work
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();

    // cv::Mat img;
    bool rgb_success = rgb_camera_->getRGBImage(image_);
    // std::cout << "CAMERA IMAGE" << std::endl;
    // std::cout << "success: " << rgb_success << " rows: " << img.rows << ", cols: " << img.cols << std::endl;
    if (rgb_success) {
      cv::imwrite("~/Desktop/flightmare_cam_test/" + std::to_string(image_counter_) + ".png", image_);
    }
    image_counter_++;
  } else {
    std::cout << "no unity render or anything like that available" << std::endl;
  }

  return true;
}

Scalar RacingTestEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  return 0.0;
}

bool RacingTestEnv::isTerminalState(Scalar &reward) {
  return false;
}

bool RacingTestEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["racing_test_env"]) {
    sim_dt_ = cfg["racing_test_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["racing_test_env"]["max_t"].as<Scalar>();
    // std::cout << "camera set in yaml: " << cfg["racing_test_env"]["camera"] << std::endl;
  } else {
    return false;
  }

  return true;
}

bool RacingTestEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool RacingTestEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void RacingTestEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

bool RacingTestEnv::setUnity(bool render) {
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

bool RacingTestEnv::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}

void RacingTestEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

std::ostream &operator<<(std::ostream &os, const RacingTestEnv &quad_env) {
  os.precision(3);
  os << "Racing Test Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib