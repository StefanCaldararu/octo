from magpie.ur5 import UR5_Interface as ur5
from magpie.ur5 import pose_vector_to_homog_coord
from magpie.ur5 import homog_coord_to_pose_vector
import time

import gym
import numpy as np
from pyquaternion import Quaternion
from magpie import realsense_wrapper as real
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import asyncio

def step_action(robot, action, blocking=True):
    # get the robot pose
    curr_pose = homog_coord_to_pose_vector(robot.getPose())
    print(curr_pose)
    print(action)
    pose_act = []
    for i in range(0,6):
        pose_act.append(curr_pose[i] + action[i])
    homog = pose_vector_to_homog_coord(pose_act)
    #TODO: do something with robot
    robot.moveL(homog, linSpeed=0.02, asynch = not blocking)
    if(action[-1] == 1.0):
        robot.open_gripper()
    elif action[-1] < 1.0:
        robot.set_gripper(0.02)

    # robot.set_gripper(0.02)

def get_observation(robot, gripper_cam, workspace_cam):
    #TODO: get the observation
    gripper_im = gripper_cam.takeImages()
    workspace_im = workspace_cam.takeImages()
    
    return np.append(robot.get_joint_angles(), robot.get_gripper_sep()), gripper_im, workspace_im
    
def null_obs(im_size):
    #TODO: implement
    return {
        "image_wrist": np.zeros((im_size, im_size, 3), dtype=np.uint8),
        "image_primary": np.zeros((im_size, im_size, 3), dtype=np.uint8),
        # "proprio": np.zeros((8,), dtype=np.float64)
    }

def convert_obs(obs_gripper_im, obs_workspace_im, obs_pose, im_size):
    gripper_im = (cv2.resize(np.asarray(obs_gripper_im.color), (im_size, im_size))).astype(np.uint8)
    workspace_im = (cv2.resize(np.asarray(obs_workspace_im.color), (im_size, im_size))).astype(np.uint8)
    image_bgr = cv2.cvtColor(gripper_im, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #TODO: implement
    return {
        "image_wrist": gripper_im,
        "image_primary": workspace_im 
        # "proprio": obs_pose
    }

def reset(robot, blocking=True):
    #TODO:
    robot.moveJ([0.643, -1.7, 1.667, -1.712, -1.47, -1.047], rotSpeed=0.1, asynch=not blocking)
    robot.open_gripper()

class UR5Gym(gym.Env):

    def __init__(
            self,
            ur5_client: ur5,
            gripper_cam: real.RealSense,
            workspace_cam: real.RealSense,
            im_size: int = 256,
            blocking: bool = True,
            sticky_gripper_num_steps: int = 1,
    ):
        self.ur5_client = ur5_client
        self.gripper_cam = gripper_cam
        self.workspace_cam = workspace_cam
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_wrist": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255*np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "image_primary": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),    
                    high=255*np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                # "pad_mask_dict/image_primary": gym.spaces.Discrete(2)
                # "proprio": gym.spaces.Box(
                #     low=np.ones((8,))*-1, high = np.ones((8,)), dtype=np.float64
                # ),
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

        raw_pose, raw_gripper_image, raw_workspace_image = get_observation(self.ur5_client, self.gripper_cam, self.workspace_cam)

        truncated = False
        if raw_pose is None or raw_gripper_image is None or raw_workspace_image is None:
            # loss of conection
            truncated = True
            obs = null_obs(self.im_size())
        else:
            obs = convert_obs(raw_gripper_image, raw_workspace_image, raw_pose, self.im_size)

        return obs, 0, False, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        reset(self.ur5_client)

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        raw_pose, raw_gripper_image, raw_workspace_image = get_observation(self.ur5_client, self.gripper_cam, self.workspace_cam)
        obs = convert_obs(raw_gripper_image, raw_workspace_image, raw_pose, self.im_size)

        return obs, {}
        