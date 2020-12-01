#include "flightlib/envs/env_base_camera.hpp"

namespace flightlib {

EnvBaseCamera::EnvBaseCamera() : obs_dim_(0), act_dim_(0), sim_dt_(0.0) {}

EnvBaseCamera::~EnvBaseCamera() {}

void EnvBaseCamera::curriculumUpdate() {}

void EnvBaseCamera::close() {}

void EnvBaseCamera::render() {}

void EnvBaseCamera::updateExtraInfo() {}

bool EnvBaseCamera::isTerminalState(Scalar &reward) {
  reward = 0.f;
  return false;
}

}  // namespace flightlib
