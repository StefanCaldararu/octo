import jax
from octo.model.octo_model import OctoModel
import cv2
from magpie.ur5 import UR5_Interface as ur5
from magpie import realsense_wrapper as real
from octo.utils.train_callbacks import supply_rng
from functools import partial

import numpy as np

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

#setup the actions
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

# get the image
raw_image = workspace_rs.takeImages()
image = (cv2.resize(np.asarray(raw_image.color), (256, 256))).astype(np.uint8)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow('Image', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
obs = {"image_primary": np.array([image,image]),
       "timestep_pad_mask": np.array([0., 1.]),
    }
task =  model.create_tasks(texts=["pick up the green block"])
action = np.array(policy_fn(obs, task), dtype = np.float64)
print(action)