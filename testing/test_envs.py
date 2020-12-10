from ruamel.yaml import YAML, dump, RoundTripDumper

#
import cv2
import os
import numpy as np
from time import time

#
from rpg_baselines.ppo.ppo2 import PPO2
from testing.test_env_wrapper import TestEnvWrapper

#
from flightgym import QuadrotorEnv_v1, TestEnv_v0, RacingTestEnv_v0


def main():
    # load the config
    config = YAML().load(open(os.path.join(os.getenv("FLIGHTMARE_PATH"), "flightlib/configs/racing_test_env.yaml"), "r"))
    # TODO: figure out mpc it complains when using this config as an input argument

    # load a model to get some decent output
    model = PPO2.load("../flightrl/examples/saved/quadrotor_env.zip")
    # model = None

    # get the environment
    env = TestEnvWrapper(RacingTestEnv_v0())
    env.connectUnity()
    observation, image = env.reset()

    action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    """
    observation, image = env.step(action)
    cv2.imwrite("/home/simon/Desktop/flightmare_cam_test/test.png", image)

    # env.test()

    exit()
    """

    writer = cv2.VideoWriter(
        "/home/simon/Desktop/flightmare_cam_test/quad_video.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        30.0,
        (800, 600)
    )

    # run loop with predictions from trained model
    start = time()
    last = start

    test = []
    c = 0
    while (time() - start) < 10:
        action, _ = model.predict(observation, deterministic=True)
        observation, image = env.step(action)

        # print(image.shape)
        writer.write(image)
        # cv2.imwrite("/home/simon/Desktop/flightmare_cam_test/test_{}.png".format(c), image)
        c += 1

        # cv2.imshow("camera", image)
        # cv2.waitKey(30)

        current = time()
        test.append(current - last)
        last = current

    writer.release()

    env.disconnectUnity()

    print("Step time mean:", np.mean(test), "\nStep time std:", np.std(test))
    print("FPS:", 1.0 / np.mean(test))


if __name__ == "__main__":
    main()
