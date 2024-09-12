from envs.magpie.ur5 import UR5_Interface as ur5
import time

import gym
import numpy as np
from pyquaternion import Quaternion
from envs.magpie import realsense_wrapper as real

def step_action(robot, action, blocking=True):
    #TODO: do something with robot
    robot.moveJ(pose = action[:7], rotSpeed=0.1, asynch = not blocking)
    robot.set_gripper(0.02)

def get_observation(robot, cam):
    #TODO: get the observation
    
    return (robot.get_joing_angles()).append(robot.get_gripper_sep()), cam.takeImages(path = "", save=False)
    
def null_obs(im_size):
    #TODO: implement
    return {
        "image_wrist": np.zeros((im_size, im_size, 3), dtype=np.uint8),
        "proprio": np.zeros((8,), dtype=np.float64)
    }

def convert_obs(obs_im, obs_pose, im_size):
    im = (obs_im.reshape(3, im_size, im_size).transpose(1,2,0) * 255).astype(np.uint8)

    #TODO: implement
    return {
        "image_wrist": im, 
        "proprio": obs_pose
    }

def reset(robot, blocking=True):
    #TODO:
    robot.moveJ(pose = [0.643, -1.7, 1.667, -1.712, -1.47, -1.047], rotSpeed=0.1, asynch=not blocking)
    robot.open_gripper()

class UR5Gym(gym.Env):

    def __init__(
            self,
            ur5_client: ur5,
            cam: real.RealSense,
            im_size: int = 256,
            blocking: bool = True,
            sticky_gripper_num_steps: int = 1,
    ):
        self.ur5_client = ur5_client
        self.cam = cam
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_wrist": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255*np.ones((im_size, im_size, 3)),
                    dtype = np.uint8,
                ),
                "proprio": gym.space.Box(
                    low=np.ones((8,))*-1, high = np.ones((8,)), dtype=np.float64
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros((7,)), high=np.ones((7,)), dtype = np.float64
        )
        self.sticky_gripper_num_steps = sticky_gripper_num_steps
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

    def step(self, action):
        # sticky gripper logic
        if(action[-1] < 0.5) != self.is_gripper_closed:
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.is_gripper_closed = not self.is_gripper_closed
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.2 if self.is_gripper_closed else 1.0
        step_action(self.ur5_client, action, blocking=self.blocking)

        raw_pose, raw_image = get_observation(self.ur5_client)

        truncated = False
        if raw_pose is None or raw_image is None:
            # loss of conection
            truncated = True
            obs = null_obs(self.im_size())
        else:
            obs = convert_obs(raw_image, raw_pose, self.im_size)

        return obs, 0, False, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        reset(self.ur5_client)

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        raw_obs = get_observation(self.ur5_client)
        obs = convert_obs(raw_obs, self.im_size)

        return obs, {}
        