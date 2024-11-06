from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
#TODO: import ur5_env
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, TemporalEnsembleWrapper
from octo.utils.train_callbacks import supply_rng

from magpie.ur5 import UR5_Interface as ur5
from magpie import realsense_wrapper as real
from envs.ur5_env import UR5Gym


logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

# custom to bridge_data_robot
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")


flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("window_size", 2, "Observation history length")
flags.DEFINE_integer(
    "action_horizon", 4, "Length of action sequence to execute/ensemble"
)

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.574, -1.511, 1.083, -1.982, -1.661, -1.824, 0], [1.22, -1.075, 1.81, -1.359, -1.604, 1.283, 0]] # TODO: Set workspace bounds
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

# def get_observation(robot, gripper_cam, workspace_cam):
#     #TODO: get the observation
#     gripper_im = gripper_cam.takeImages()
#     workspace_im = workspace_cam.takeImages()
    
#     return np.append(robot.get_joint_angles(), robot.get_gripper_sep()), gripper_im, workspace_im

# def convert_obs(obs_gripper_im, obs_workspace_im, obs_pose, im_size):
#     gripper_im = (cv2.resize(np.asarray(obs_gripper_im.color), (im_size, im_size))).astype(np.uint8)
#     workspace_im = (cv2.resize(np.asarray(obs_workspace_im.color), (im_size, im_size))).astype(np.uint8)
#     image_bgr = cv2.cvtColor(gripper_im, cv2.COLOR_RGB2BGR)
#     cv2.imshow('Image', image_bgr)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     #TODO: implement
#     return {
#         "image_wrist": gripper_im,
#         "image_primary": workspace_im 
#         # "proprio": obs_pose
#     }

# def step_action(robot, action, blocking=True):
#     # get the robot pose
#     curr_pose = homog_coord_to_pose_vector(robot.getPose())
#     print(curr_pose)
#     print(action)
#     pose_act = []
#     for i in range(0,6):
#         pose_act.append(curr_pose[i] + action[i])
#     homog = pose_vector_to_homog_coord(pose_act)
#     #TODO: do something with robot
#     robot.moveL(homog, linSpeed=0.02, asynch = not blocking)
#     if(action[-1] == 1.0):
#         robot.open_gripper()
#     elif action[-1] < 1.0:
#         robot.set_gripper(0.02)



def main(_):
    # load models
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

    #initialize cam and robot
    robot = ur5()
    robot.start()
    # robot.moveJ([0.643, -1.7, 1.667, -1.712, -1.47, -1.047], rotSpeed=0.1, asynch=False)
    # rsc = real.RealSense()
    # rsc.initConnection()
    devices = real.poll_devices()
    wrist_rs = real.RealSense(fps=5, device_name='D405')
    wrist_rs.initConnection(device_serial=devices['D405'])
    devices = real.poll_devices()
    workspace_rs = real.RealSense(zMax=5, fps=6, device_name='D435')
    workspace_rs.initConnection(device_serial=devices['D435'])
    wrist_rs.flush_buffer(2)
    workspace_rs.flush_buffer(2)
    #wrap the robot environment
    env = UR5Gym(robot, wrist_rs, workspace_rs, FLAGS.im_size)
    env = HistoryWrapper(env, FLAGS.window_size)
    env = TemporalEnsembleWrapper(env, FLAGS.action_horizon)

    def sample_actions(
            pretrained_model: OctoModel,
            observations,
            tasks,
            rng,
    ):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
        )
        # remove batch dim
        return actions[0]
    
    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
        )
    )

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = "Pick up the green block."
    task = model.create_tasks(texts=[goal_instruction])
    #TODO: actually get observations
    # obs = np.array([])
    print(goal_instruction)
    time.sleep(2.0)

    # null_observation = {
    #     "image_wrist": np.zeros((FLAGS.im_size, im_size, 3), dtype=np.uint8),
    #     "image_primary": np.zeros((im_size, im_size, 3), dtype=np.uint8),
    # }

    last_tstep = time.time()
    images = []
    goals = []
    t = 0
    obs, something = env.reset()
    while t < FLAGS.num_timesteps:
        if time.time() > last_tstep + STEP_DURATION:
            last_tstep = time.time()
            images.append(obs["image_wrist"][-1])
            goals.append(goal_image)
            print("THE OBSERVATION IS: \n\n")
            print(obs)
            # get action
            forward_pass_time = time.time()
            action = np.array(policy_fn(obs, task), dtype = np.float64)
            print("THE ACTION IS: \n\n")
            print(action)
            action = np.array([
                [0.1,0,0,0,0,0,1.0],
                [0.1,0,0,0,0,0,1.0],
                [0.1,0,0,0,0,0,1.0],
                [0.1,0,0,0,0,0,1.0]
            ])
            # perform the step
            obs = env.step(action)
    robot.stop()
            
if __name__ == "__main__":
    app.run(main)
