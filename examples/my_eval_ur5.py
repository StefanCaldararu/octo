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


logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)

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
# WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]] # TODO: Set workspace bounds
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

def main(_):
    # load models
    model = OctoModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

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
            unnormalization_statistics=pretrained_model.dataset_statistics[
                "bridge_dataset"
            ]["action"],
        )
        # remove batch dim
        return actions[0]
    
    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            argmax=FLAGS.deterministic,
            temperature=FLAGS.temperature,
        )
    )

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = "Pick up the blue block."

    task = model.create_tasks(texts=[goal_instruction])
    #TODO: actually get observations
    obs = np.array([])

    time.sleep(2.0)

    last_tstep = time.time()
    images = []
    goals = []
    t = 0
    while t < FLAGS.num_timesteps:
        if time.time() > last_tstep + STEP_DURATION:
            last_tstep = time.time()
            images.append(obs["image_primary"][-1])
            goals.append(goal_image)
            
            # get action
            forward_pass_time = time.time()
            action = np.array(policy_fn(obs, task), dtype = np.float64)
            
            # perform the step
            
