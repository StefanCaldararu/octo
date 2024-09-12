from magpie.ur5 import UR5_Interface as ur5
import time

import gym
import numpy as np
from pyquaternion import Quaternion
from magpie import realsense_wrapper as real

def step_action(robot, action, blocking=True):
    #TODO: do something with robot
    robot.close_gripper()

def get_observation(robot):
    #TODO: get the observation
    robot.open_gripper()
    
def null_obs(im_size):
    #TODO: implement
    return None

def convert_obs(obs, im_size):
    #TODO: implement
    return None

def reset(robot, blocking=True):
    #TODO:
    robot.moveJ(pose = [0.643, -1.7, 1.667, -1.712, -1.47, -1.047], rotSpeed=0.1, asynch=not blocking)
    robot.open_gripper()

class UR5Gym(gym.Env):

    def __init__(
            self,
            ur5_client: ur5,
            im_size: int = 256,
            blocking: bool = True,
            sticky_gripper_num_steps: int = 1,
    ):
        self.ur5_client = ur5_client
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

        raw_obs = get_observation(self.ur5_client)

        truncated = False
        if raw_obs is None:
            # loss of conection
            truncated = True
            obs = null_obs(self.im_size())
        else:
            obs = convert_obs(raw_obs, self.im_size)

        return obs, 0, False, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        reset(self.ur5_client)

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        raw_obs = get_observation(self.ur5_client)
        obs = convert_obs(raw_obs, self.im_size)

        return obs, {}