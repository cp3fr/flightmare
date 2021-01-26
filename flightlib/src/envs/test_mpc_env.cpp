#include "flightlib/envs/test_mpc_env.hpp"

namespace flightlib {

MPCTest::MPCTest() : MPCTest(getenv("FLIGHTMARE_PATH") + std::string("/flightlib/configs/quadrotor_env.yaml")) {}

MPCTest::MPCTest(const std::string &cfg_path, const bool wave_track) {
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
  world_box_ << -30, 30, -30, 30, 0, 30;
  quadrotor_ptr_->setWorldBox(world_box_);

  // add dummy cameras
  /*
  Vector<3> dummy_translation(0.0, 0.5, 0.3);
  Matrix<3, 3> dummy_rotation = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  for (int i = 0; i < 3; i++) {
    camera_dummy_[i] = std::make_shared<Quadrotor>();
    camera_dummy_[i]->setWorldBox(world_box_);

    camera_dummy_state_[i].x = Vector<25>::Zero();
    camera_dummy_state_[i].t = (Scalar) 0.0f;
    camera_dummy_state_[i].x.segment<3>(0) << 0.0, -15.0, ((Scalar) i) * 3.0;
    camera_dummy_state_[i].x.segment<2>(3) << std::cos(M_PI_2 * -0.25), std::sin(M_PI_2 * -0.25);
    camera_dummy_[i]->setState(camera_dummy_state_[i]);

    dummy_camera_[i] = std::make_unique<RGBCamera>();
    dummy_camera_[i]->setFOV(mpcenv::fov);
    dummy_camera_[i]->setHeight(mpcenv::image_height);
    dummy_camera_[i]->setWidth(mpcenv::image_width);
    dummy_camera_[i]->setRelPose(dummy_translation, dummy_rotation);
    dummy_camera_[i]->setPostProcesscing(std::vector<bool>{false, false, false});
    camera_dummy_[i]->addRGBCamera(dummy_camera_[i]);
  }*/

  /*
  camera_dummy_ = std::make_shared<Quadrotor>();
  camera_dummy_->setWorldBox(world_box_);

  camera_dummy_state_.x = Vector<25>::Zero();
  camera_dummy_state_.t = (Scalar) 0.0f;
  camera_dummy_state_.x.segment<3>(0) << 0.0, -15.0, 7.0;
  camera_dummy_state_.x.segment<2>(3) << std::cos(M_PI_2 * -0.25), std::sin(M_PI_2 * -0.25);
  camera_dummy_->setState(camera_dummy_state_);

  Vector<3> B_r_BC(0.0, 0.5, 0.3);
  Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
  */

  // own tests
  // Vector<3> B_r_BC(0.5, 0.0, 0.3);
  // Matrix<3, 3> R_BC = Quaternion(std::cos(-0.5 * M_PI_2), 0.0, 0.0, std::sin(-0.5 * M_PI_2)).toRotationMatrix();

  // airsim
  float uptilt_angle = 30.0;
  uptilt_angle = -(uptilt_angle / 90.0) * M_PI_2;
  Vector<3> B_r_BC(0.2, 0.0, 0.1);
  // Vector<3> B_r_BC(0.2, 0.032, 0.1);
  // Matrix<3, 3> R_BC = Quaternion(std::cos(0.5 * uptilt_angle), std::sin(0.5 * uptilt_angle), 0.0, 0.0).toRotationMatrix();
  Matrix<3, 3> R_BC = Quaternion(std::cos(0.5 * uptilt_angle), 0.0, std::sin(0.5 * uptilt_angle), 0.0).toRotationMatrix();
  Matrix<3, 3> temp = Quaternion(std::cos(-0.5 * M_PI_2), 0.0, 0.0, std::sin(-0.5 * M_PI_2)).toRotationMatrix();
  R_BC = R_BC * temp;

  rgb_camera_ = std::make_unique<RGBCamera>();
  image_counter_ = 0;
  rgb_camera_->setFOV(mpcenv::fov);
  rgb_camera_->setHeight(mpcenv::image_height);
  rgb_camera_->setWidth(mpcenv::image_width);
  rgb_camera_->setRelPose(B_r_BC, R_BC);
  rgb_camera_->setPostProcesscing(std::vector<bool>{false, false, false});
  quadrotor_ptr_->addRGBCamera(rgb_camera_);

  // add gates, hard-coded for now
  for (int i = 0; i < mpcenv::num_gates; i++) {
    gates_[i] = std::make_shared<StaticGate>("test_gate_" + std::to_string(i), "rpg_gate");
    gates_[i]->setPosition(Eigen::Vector3f(POSITIONS[i][0], POSITIONS[i][1], POSITIONS[i][2]));
    gates_[i]->setRotation(Quaternion(std::cos(ORIENTATIONS[i]), 0.0, 0.0, std::sin(ORIENTATIONS[i])));
  }
  setWaveTrack(wave_track);

  std::cout << "Gate size: " << gates_[0]->getSize().transpose() << std::endl;

  // add unity
  setUnity(true);
}

MPCTest::~MPCTest() {}

bool MPCTest::step(Ref<Vector<>> new_state, Ref<ImageFlat<>> image) {
  quad_state_.x.segment<10>(0) = new_state;  // should maybe express this as sum instead of fixed number
  quadrotor_ptr_->setState(quad_state_);
  // std::cout << "new_state: " << new_state.transpose() << std::endl;
  // std::cout << "quad_state_.x: " << quad_state_.x.transpose() << std::endl << std::endl;

  // also capture an image
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(0);
    unity_bridge_ptr_->handleOutput();

    bool rgb_success = rgb_camera_->getRGBImage(cv_image_);

    if (rgb_success) {
      cv::split(cv_image_, cv_channels_);
      for (int i = 0; i < cv_image_.channels(); i++) {
        cv::cv2eigen(cv_channels_[i], channels_[i]);
        Map<ImageFlat<>> image_(channels_[i].data(), channels_[i].size());
        image.block<mpcenv::image_height * mpcenv::image_width, 1>(i * mpcenv::image_height * mpcenv::image_width, 0) = image_;
      }
    }
  } else {
    std::cout << "Unity rendering not available; cannot get images." << std::endl;
  }

  return true;
}

void MPCTest::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  // bridge->addQuadrotor(camera_dummy_);
  bridge->addQuadrotor(quadrotor_ptr_);
  for (int i = 0; i < mpcenv::num_gates; i++) {
    bridge->addStaticObject(gates_[i]);
  }
  /*
  for (int i = 0; i < 3; i++) {
    bridge->addQuadrotor(camera_dummy_[i]);
  }
  */
}

bool MPCTest::setUnity(bool render) {
  unity_render_ = render;
  if (unity_render_ && unity_bridge_ptr_ == nullptr) {
    // create unity bridge
    unity_bridge_ptr_ = UnityBridge::getInstance();
    // add this environment to Unity
    this->addObjectsToUnity(unity_bridge_ptr_);
  }
  return true;
}

bool MPCTest::connectUnity(const int pub_port, const int sub_port) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_, pub_port, sub_port);
  return unity_ready_;
}

void MPCTest::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    std::cout << "WARNING: Flightmare Unity Bridge is not initialized." << std::endl;
  }
}

int MPCTest::getImageHeight() const {
  return mpcenv::image_height;
}

int MPCTest::getImageWidth() const {
  return mpcenv::image_width;
}

void MPCTest::setWaveTrack(bool wave_track) {
  float pos_z;
  int i;
  for (int j = 0; j < mpcenv::num_elevated_gates; j++) {
    i = ELEVATED_GATES_INDICES[j];
    pos_z = POSITIONS[i][2];
    if (wave_track) {
      pos_z += 3.0;
    }
    std::cout << pos_z << std::endl;
    gates_[i]->setPosition(Eigen::Vector3f(POSITIONS[i][0], POSITIONS[i][1], pos_z));
  }
}

bool MPCTest::loadParam(const YAML::Node &cfg) {
  if (cfg["test"]) {
    // load reinforcement learning related parameters
    test_yaml_ = cfg["test"]["test_yaml"].as<std::vector<std::vector<Scalar>>>();

    Eigen::MatrixXf test(3, 3);
    for (int i = 0; i < 3; i++) {
      test.row(i) = Eigen::VectorXf::Map(&test_yaml_[i][0], test_yaml_[i].size());
    }

    std::cout << "YAML test:" << std::endl << test << std::endl;
  } else {
    return false;
  }

  return true;
}

}  // namespace flightlib