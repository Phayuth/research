import os
import numpy as np
import matplotlib.pyplot as plt
from spatial_geometry.utils import Utils
from robot.nonmobile.planar_rr import PlanarRR

# robot
robot = PlanarRR()

# init
limt2 = np.array([[-2 * np.pi, 2 * np.pi], [-2 * np.pi, 2 * np.pi]])

q1 = np.array([-6, 5]).reshape(2, 1)
q2 = np.array([6, 5]).reshape(2, 1)
q3 = np.array([-5, 6]).reshape(2, 1)
q4 = np.array([5, -6]).reshape(2, 1)

fk1 = robot.forward_kinematic(q1)
fk2 = robot.forward_kinematic(q2)
fk3 = robot.forward_kinematic(q3)
fk4 = robot.forward_kinematic(q4)
# fk1 = np.array([0.3, 1]).reshape(2, 1)
# fk2 = np.array([1, 1.5]).reshape(2, 1)
# fk3 = np.array([1.5, 1]).reshape(2, 1)
# fk4 = np.array([1.5, 1.5]).reshape(2, 1)

ik11 = robot.inverse_kinematic_geometry(fk1, elbow_option=1)
ik12 = robot.inverse_kinematic_geometry(fk1, elbow_option=0)
ik21 = robot.inverse_kinematic_geometry(fk2, elbow_option=1)
ik22 = robot.inverse_kinematic_geometry(fk2, elbow_option=0)
ik31 = robot.inverse_kinematic_geometry(fk3, elbow_option=1)
ik32 = robot.inverse_kinematic_geometry(fk3, elbow_option=0)
ik41 = robot.inverse_kinematic_geometry(fk4, elbow_option=1)
ik42 = robot.inverse_kinematic_geometry(fk4, elbow_option=0)

qalt11 = Utils.find_alt_config(ik11.reshape(2, 1), limt2)
qalt12 = Utils.find_alt_config(ik12.reshape(2, 1), limt2)
qalt21 = Utils.find_alt_config(ik21.reshape(2, 1), limt2)
qalt22 = Utils.find_alt_config(ik22.reshape(2, 1), limt2)
qalt31 = Utils.find_alt_config(ik31.reshape(2, 1), limt2)
qalt32 = Utils.find_alt_config(ik32.reshape(2, 1), limt2)
qalt41 = Utils.find_alt_config(ik41.reshape(2, 1), limt2)
qalt42 = Utils.find_alt_config(ik42.reshape(2, 1), limt2)

# cspace collision point
rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"
collision = np.load(rsrc + "collisionpoint_exts.npy")


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.set_aspect("equal")
ax.set_xlim([-2 * np.pi, 2 * np.pi])
ax.set_ylim([-2 * np.pi, 2 * np.pi])
ax.set_xlabel(r"$q_1$")
ax.set_ylabel(r"$q_2$")
ax.set_xticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
ax.set_xticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
ax.set_yticks([-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi])
ax.set_yticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
ax.grid(True, linestyle="--", alpha=0.5)

# ax.plot(collision[:, 0], collision[:, 1], "k.", markersize=1, alpha=0.5)

ax.plot(qalt11[0, :], qalt11[1, :], "ro", label="ik11")
ax.plot(qalt12[0, :], qalt12[1, :], "r^", label="ik12")
ax.plot(qalt21[0, :], qalt21[1, :], "go", label="ik21")
ax.plot(qalt22[0, :], qalt22[1, :], "g^", label="ik22")
ax.plot(qalt31[0, :], qalt31[1, :], "bo", label="ik31")
ax.plot(qalt32[0, :], qalt32[1, :], "b^", label="ik32")
ax.plot(qalt41[0, :], qalt41[1, :], "mo", label="ik41")
ax.plot(qalt42[0, :], qalt42[1, :], "m^", label="ik42")
ax.legend()


plt.show()
