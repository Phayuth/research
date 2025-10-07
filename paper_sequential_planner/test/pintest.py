import numpy as np
import pinocchio

model = pinocchio.buildSampleModelManipulator()
data = model.createData()

q = pinocchio.neutral(model)
pinocchio.forwardKinematics(model, data, q)
for i, frame in enumerate(model.frames):
    print(f"Frame {i}: {frame.name}")
pinocchio.updateFramePlacements(model, data)

ee_frame_id = model.getFrameId("effector_body")  # name depends on model
ee_placement = data.oMf[ee_frame_id]
print("End-effector placement:", ee_placement)

JOINT_ID = 6
J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
