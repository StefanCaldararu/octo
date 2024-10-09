from magpie import realsense_wrapper
import asyncio
import matplotlib.pyplot as plt
devices = realsense_wrapper.poll_devices()

wrist_rs = realsense_wrapper.RealSense(fps=5, device_name='D405')
wrist_rs.initConnection(device_serial=devices['D405'])
devices = realsense_wrapper.poll_devices()
workspace_rs = realsense_wrapper.RealSense(zMax=5, fps=6, device_name='D435')
workspace_rs.initConnection(device_serial=devices['D435'])

wrist_rs.flush_buffer(2)
workspace_rs.flush_buffer(2)
async def im():
    color = await wrist_rs.take_image()
    color2 = await workspace_rs.take_image()
    plt.imshow(color)
    plt.show()
    plt.imshow(color2)
    plt.show()

asyncio.run(im())