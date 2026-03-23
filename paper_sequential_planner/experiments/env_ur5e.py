import os
import tqdm
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from spatial_geometry.utils import Utils
from scipy.spatial.transform import Rotation as R

try:
    from eaik.IK_DH import DhRobot
    from spatialmath import SE3
    from roboticstoolbox import DHRobot, RevoluteDH
except:
    print("missing packages; usage limited")

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    ou.RNG.setSeed(42)
    ompl_available = True
except ImportError:
    print("OMPL not available, limitted functionality without OMPL.")
    ompl_available = False

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class RobotUR5eKin:

    def __init__(self):
        self.d = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
        self.alpha = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])
        self.a = np.array([0, -0.425, -0.3922, 0, 0, 0])
        self.qlim_dict = {
            "hardware_default": np.array(
                [
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-np.pi, np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                ]
            ),
            "table_under": np.array(
                [
                    [-2 * np.pi, 2 * np.pi],
                    [-np.pi, np.pi],
                    [-np.pi, np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                    [-2 * np.pi, 2 * np.pi],
                ]
            ),
        }

        Ls = [
            RevoluteDH(
                d=self.d[i],
                a=self.a[i],
                alpha=self.alpha[i],
                qlim=self.qlim_dict["hardware_default"][i],
            )
            for i in range(6)
        ]

        # bot kinematics
        self.bot = DhRobot(self.alpha, self.a, self.d)
        self.bot_rtb = DHRobot(Ls, name="UR5e")

    def solve_fk(self, q):
        return self.bot.fwdKin(q)

    def solve_aik(self, H):
        sols = self.bot.IK(H)
        numsols = sols.num_solutions()
        Q = sols.Q
        return numsols, Q

    def solve_nik(self, H, q0):
        sol = self.bot_rtb.ikine_LM(H, q0=q0)
        return sol.q

    def solve_aik_bulk(self, H):
        num_sols = []
        ik_sols = []
        for h in H:
            num_sol, ik_sol = self.solve_aik(h)
            num_sols.append(num_sol)
            ik_sols.append(ik_sol)
        return num_sols, ik_sols

    def solve_aik_altconfig(self, H):
        numsol, Qik = self.solve_aik(H)
        limt6 = self.qlim_dict["hardware_default"]
        ikaltconfig = []
        for i in range(numsol):
            alt = Utils.find_alt_config(Qik[i].reshape(6, 1), limt6)
            ikaltconfig.append(alt)
        ikaltconfig = np.hstack(ikaltconfig).T
        return ikaltconfig.shape[0], ikaltconfig

    def solve_aik_altconfig_bulk(self, H):
        num_sols = []
        ik_sols = []
        for h in H:
            num_sol, ik_solaltconfig = self.solve_aik_altconfig(h)
            num_sols.append(num_sol)
            ik_sols.append(ik_solaltconfig)
        return num_sols, ik_sols

    def solve_manipulability(self, q):
        return self.bot_rtb.manipulability(q)

    def solve_jacobian(self, q):
        return self.bot_rtb.jacob0(q)

    def _convert_urdf_to_dh_frame(H):
        "from our design task in urdf frame to dh frame"
        Hdh_to_urdf = SE3.Rz(np.pi).A
        return np.linalg.inv(Hdh_to_urdf) @ H

    def _convert_dh_to_urdf_frame(H):
        "from dh frame to our design task in urdf frame"
        Hdh_to_urdf = SE3.Rz(np.pi).A
        return Hdh_to_urdf @ H


class Constants:
    rsrcpath = os.environ["RSRC_DIR"] + "/rnd_torus/"
    ur5e_urdf = rsrcpath + "ur5e/ur5e_extract_calibrated.urdf"
    plane_urdf = "plane.urdf"
    pole_urdf = rsrcpath + "ur5e/simple_box.urdf"
    table_urdf = "table/table.urdf"
    shelf_urdf = rsrcpath + "ur5e/shelf_texture.urdf"

    camera_exp3 = (
        1.0,
        61.20,
        -42.20,
        (-0.15094073116779327, 0.1758367419242859, 0.10792634636163712),
    )

    model_list = [
        (table_urdf, [0, 0, 0], [0, 0, 0, 1]),
        (pole_urdf, [0.3, 0.3, 0], [0, 0, 0, 1]),
        (pole_urdf, [-0.3, 0.3, 0], [0, 0, 0, 1]),
        (pole_urdf, [-0.3, -0.3, 0], [0, 0, 0, 1]),
        (pole_urdf, [0.3, -0.3, 0], [0, 0, 0, 1]),
    ]

    model_list_shelf = [
        (shelf_urdf, [0, 0.75, 0], [0, 0, 1, 0]),
        (shelf_urdf, [0, -0.75, 0], [0, 0, 0, 1]),
        (shelf_urdf, [-0.75, 0, 0], [0, 0, -0.5, 0.5]),
    ]


class UR5eBullet:

    def __init__(self, mode="gui") -> None:
        # connect
        if mode == "gui":
            p.connect(p.GUI)
            # p.connect(p.SHARED_MEMORY_GUI)
        if mode == "no_gui":
            p.connect(p.DIRECT)
            # p.connect(p.SHARED_MEMORY)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # load model and properties
        self.load_model()
        self.models_collision = [self.planeID]
        self.models_others = []
        self.ghost_model = []
        self.numJoints = self.get_num_joints()
        self.jointNames = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.jointIDs = [1, 2, 3, 4, 5, 6]
        self.gripperlinkid = 9

        # inverse kinematic
        self.lower_limits = [-np.pi] * 6
        self.upper_limits = [np.pi] * 6
        self.joint_ranges = [2 * np.pi] * 6
        self.rest_poses = [0, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self.joint_damp = [0.01] * 6

        # color for joint
        self.axis_len = 0.25
        self.line_width = 2

    def load_model(self):
        # self.draw_frame([0, 0, 0], [0, 0, 0, 1], length=1, width=3)
        self.planeID = p.loadURDF(Constants.plane_urdf, [0, 0, -0.02])
        self.ur5eID = p.loadURDF(
            Constants.ur5e_urdf,
            [0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        return self.ur5eID, self.planeID

    def load_models_other(self, model_list):
        for murdf, mp, mo in model_list:
            mid = p.loadURDF(murdf, mp, mo, useFixedBase=True)
            self.models_others.append(mid)
        self.models_collision.extend(self.models_others)
        return self.models_others

    def load_models_ghost(self, color=None):
        ur5e_ghost_id = p.loadURDF(
            Constants.ur5e_urdf,
            [0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        # Disable all collisions
        for link_index in range(-1, p.getNumJoints(ur5e_ghost_id)):
            p.setCollisionFilterGroupMask(ur5e_ghost_id, link_index, 0, 0)

        # Set the color of the ghost model
        if color:
            self.change_color(
                ur5e_ghost_id,
                red=color[0],
                green=color[1],
                blue=color[2],
                alpha=color[3],
            )

        self.ghost_model.append(ur5e_ghost_id)
        return self.ghost_model

    def load_slider(self):
        self.redSlider = p.addUserDebugParameter("red", 0, 1, 1)
        self.greenSlider = p.addUserDebugParameter("green", 0, 1, 0)
        self.blueSlider = p.addUserDebugParameter("blue", 0, 1, 0)
        self.alphaSlider = p.addUserDebugParameter("alpha", 0, 1, 0.5)

        self.j1s = p.addUserDebugParameter("joint_1", -np.pi, np.pi, 0)
        self.j2s = p.addUserDebugParameter("joint_2", -np.pi, np.pi, 0)
        self.j3s = p.addUserDebugParameter("joint_3", -np.pi, np.pi, 0)
        self.j4s = p.addUserDebugParameter("joint_4", -np.pi, np.pi, 0)
        self.j5s = p.addUserDebugParameter("joint_5", -np.pi, np.pi, 0)
        self.j6s = p.addUserDebugParameter("joint_6", -np.pi, np.pi, 0)

    def read_slider(self):
        red = p.readUserDebugParameter(self.redSlider)
        green = p.readUserDebugParameter(self.greenSlider)
        blue = p.readUserDebugParameter(self.blueSlider)
        alpha = p.readUserDebugParameter(self.alphaSlider)

        j1 = p.readUserDebugParameter(self.j1s)
        j2 = p.readUserDebugParameter(self.j2s)
        j3 = p.readUserDebugParameter(self.j3s)
        j4 = p.readUserDebugParameter(self.j4s)
        j5 = p.readUserDebugParameter(self.j5s)
        j6 = p.readUserDebugParameter(self.j6s)
        return (red, green, blue, alpha), (j1, j2, j3, j4, j5, j6)

    def get_visualizer_camera(self):
        (
            width,
            height,
            viewMatrix,
            projectionMatrix,
            cameraUp,
            cameraForward,
            horizontal,
            vertical,
            yaw,
            pitch,
            dist,
            target,
        ) = p.getDebugVisualizerCamera()

        print(f"> width: {width}")
        print(f"> height: {height}")
        print(f"> viewMatrix: {viewMatrix}")
        print(f"> projectionMatrix: {projectionMatrix}")
        print(f"> cameraUp: {cameraUp}")
        print(f"> cameraForward: {cameraForward}")
        print(f"> horizontal: {horizontal}")
        print(f"> vertical: {vertical}")
        print(f"> yaw: {yaw}")
        print(f"> pitch: {pitch}")
        print(f"> dist: {dist}")
        print(f"> target: {target}")

    def set_visualizer_camera(
        self,
        cameraDistance=3,
        cameraYaw=30,
        cameraPitch=52,
        cameraTargetPosition=[0, 0, 0],
    ):
        p.resetDebugVisualizerCamera(
            cameraDistance=cameraDistance,
            cameraYaw=cameraYaw,
            cameraPitch=cameraPitch,
            cameraTargetPosition=cameraTargetPosition,
        )

    def draw_frame(self, pos, quat, length=1, width=3, text=None):
        rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]

        p.addUserDebugLine(
            pos,
            pos + length * x_axis,
            [1, 0, 0],
            lineWidth=width,
        )
        p.addUserDebugLine(
            pos,
            pos + length * y_axis,
            [0, 1, 0],
            lineWidth=width,
        )
        p.addUserDebugLine(
            pos,
            pos + length * z_axis,
            [0, 0, 1],
            lineWidth=width,
        )
        if text is not None:
            p.addUserDebugText(
                text,
                pos,
                textColorRGB=[1, 0, 1],
                textSize=1,
            )

    def change_color(self, urdf_id, red=1, green=0, blue=0, alpha=1):
        num_joints = p.getNumJoints(urdf_id)
        for link_index in range(-1, num_joints):  # -1 includes the base link
            visuals = p.getVisualShapeData(urdf_id)
            for visual in visuals:
                if visual[1] == link_index:
                    p.changeVisualShape(
                        objectUniqueId=urdf_id,
                        linkIndex=link_index,
                        rgbaColor=[red, green, blue, alpha],
                    )

    def link_viewer(self):
        for j in range(self.numJoints):
            link_state = p.getLinkState(self.ur5eID, j)[0]
            pos = link_state
            axis_len = 0.2
            p.addUserDebugLine(
                pos, [pos[0] + axis_len, pos[1], pos[2]], [0, 1, 1], 2
            )
            p.addUserDebugLine(
                pos, [pos[0], pos[1] + axis_len, pos[2]], [1, 0, 1], 2
            )
            p.addUserDebugLine(
                pos, [pos[0], pos[1], pos[2] + axis_len], [1, 1, 0], 2
            )

    def get_joint_line_color(self):
        joint_pos = self.get_array_joint_positions()
        color_list = []
        for i in range(len(joint_pos)):
            norm = min(abs(joint_pos[i]) / (2 * np.pi), 1.0)
            cr = norm
            cg = 1 - norm
            cb = 0
            color_list.append([cr, cg, cb])
        return color_list

    def get_joint_line(self):
        pos_list = []
        end_pos_list = []
        for j in self.jointIDs:
            info = p.getJointInfo(self.ur5eID, j)
            joint_axis_local = info[13]

            link_state = p.getLinkState(
                self.ur5eID, j, computeForwardKinematics=True
            )
            joint_pos_world = link_state[4]
            joint_orn_world = link_state[5]

            # Transform axis from local → world frame
            axis_world, _ = p.multiplyTransforms(
                [0, 0, 0], joint_orn_world, joint_axis_local, [0, 0, 0, 1]
            )
            axis_world = np.array(axis_world) / (np.linalg.norm(axis_world) + 1e-9)

            end_pos = [
                joint_pos_world[0] + axis_world[0] * self.axis_len,
                joint_pos_world[1] + axis_world[1] * self.axis_len,
                joint_pos_world[2] + axis_world[2] * self.axis_len,
            ]

            pos_list.append(joint_pos_world)
            end_pos_list.append(end_pos)
        return pos_list, end_pos_list

    def joint_viewer(self):
        pos_list, end_pos_list = self.get_joint_line()
        color_list = self.get_joint_line_color()
        if hasattr(self, "joint_line"):
            for pos, end_pos, color, jline in zip(
                pos_list,
                end_pos_list,
                color_list,
                self.joint_line,
            ):
                p.addUserDebugLine(
                    pos,
                    end_pos,
                    color,
                    self.line_width,
                    replaceItemUniqueId=jline,
                )
        else:
            self.joint_line = []
            for pos, end_pos, color in zip(
                pos_list,
                end_pos_list,
                color_list,
            ):
                jline = p.addUserDebugLine(
                    pos,
                    end_pos,
                    color,
                    self.line_width,
                )
                self.joint_line.append(jline)

    def get_num_joints(self):
        return p.getNumJoints(self.ur5eID)

    def get_joint_link_info(self):
        for i in range(self.numJoints):
            (
                jointIndex,
                jointName,
                jointType,
                qIndex,
                uIndex,
                flags,
                jointDamping,
                jointFriction,
                jointLowerLimit,
                jointUpperLimit,
                jointMaxForce,
                jointMaxVelocity,
                linkName,
                jointAxis,
                parentFramePos,
                parentFrameOrn,
                parentIndex,
            ) = p.getJointInfo(self.ur5eID, i)

            print(f"> ---------------------------------------------<")
            print(f"> jointIndex: {jointIndex}")
            print(f"> jointName: {jointName}")
            print(f"> jointType: {jointType}")
            print(f"> qIndex: {qIndex}")
            print(f"> uIndex: {uIndex}")
            print(f"> flags: {flags}")
            print(f"> jointDamping: {jointDamping}")
            print(f"> jointFriction: {jointFriction}")
            print(f"> jointLowerLimit: {jointLowerLimit}")
            print(f"> jointUpperLimit: {jointUpperLimit}")
            print(f"> jointMaxForce: {jointMaxForce}")
            print(f"> jointMaxVelocity: {jointMaxVelocity}")
            print(f"> linkName: {linkName}")
            print(f"> jointAxis: {jointAxis}")
            print(f"> parentFramePos: {parentFramePos}")
            print(f"> parentFrameOrn: {parentFrameOrn}")
            print(f"> parentIndex: {parentIndex}")

    def control_single_motor(self, jointIndex, jointPosition, jointVelocity=0):
        p.setJointMotorControl2(
            bodyIndex=self.ur5eID,
            jointIndex=jointIndex,
            controlMode=p.POSITION_CONTROL,
            targetPosition=jointPosition,
            targetVelocity=jointVelocity,
            positionGain=0.03,
        )

    def control_array_motors(
        self, jointPositions, jointVelocities=[0, 0, 0, 0, 0, 0]
    ):
        p.setJointMotorControlArray(
            bodyIndex=self.ur5eID,
            jointIndices=self.jointIDs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=jointPositions,
            targetVelocities=jointVelocities,
            positionGains=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        )

    def get_single_joint_state(self):
        (
            jointPosition,
            jointVelocity,
            jointReactionForce,
            appliedJointMotorTorque,
        ) = p.getJointState(self.ur5eID, jointIndex=1)
        return (
            jointPosition,
            jointVelocity,
            jointReactionForce,
            appliedJointMotorTorque,
        )

    def get_array_joint_state(self):
        j1, j2, j3, j4, j5, j6 = p.getJointStates(
            self.ur5eID, jointIndices=self.jointIDs
        )
        return j1, j2, j3, j4, j5, j6

    def get_array_joint_positions(self):
        j1, j2, j3, j4, j5, j6 = self.get_array_joint_state()
        return (j1[0], j2[0], j3[0], j4[0], j5[0], j6[0])

    def forward_kin(self):
        (
            link_trn,
            link_rot,
            com_trn,
            com_rot,
            frame_pos,
            frame_rot,
            link_vt,
            link_vr,
        ) = p.getLinkState(
            self.ur5eID,
            self.gripperlinkid,
            computeLinkVelocity=True,
            computeForwardKinematics=True,
        )
        return link_trn, link_rot

    def inverse_kin(self, positions, quaternions):
        joint_angles = p.calculateInverseKinematics(
            self.ur5eID,
            self.gripperlinkid,
            positions,
            quaternions,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            jointDamping=self.joint_damp,
            restPoses=self.rest_poses,
        )
        return joint_angles

    def collisioncheck(self):
        p.performCollisionDetection()

    def contact_point(self, modelid):
        contact_points = p.getContactPoints(
            bodyA=self.ur5eID,
            bodyB=modelid,
        )
        return contact_points

    def closest_point(self, modelid):
        closest_points = p.getClosestPoints(
            bodyA=self.ur5eID,
            bodyB=modelid,
            distance=0.5,
        )
        return closest_points

    def collsion_check(self, modelid):
        self.collisioncheck()
        cp = self.contact_point(modelid)
        if len(cp) > 0:
            return True
        else:
            return False

    def collision_check_all(self):
        self.collisioncheck()
        for mid in self.models_collision:
            cp = self.contact_point(mid)
            if len(cp) > 0:
                return True
        return False

    def collision_check_at_config(self, q):
        self.reset_array_joint_state(q)
        self.collisioncheck()
        for mid in self.models_collision:
            cp = self.contact_point(mid)
            if len(cp) > 0:
                return True
        return False

    def reset_array_joint_state(self, targetValues):
        for i in range(6):
            p.resetJointState(
                self.ur5eID,
                jointIndex=self.jointIDs[i],
                targetValue=targetValues[i],
            )

    def reset_array_joint_state_ghost(self, targetValues, urdf_id):
        for i in range(6):
            p.resetJointState(
                urdf_id,
                jointIndex=self.jointIDs[i],
                targetValue=targetValues[i],
            )

    def collision_dataset(self):
        model_id = self.load_models_other(Constants.model_list_shelf)
        num_sample = 10
        q1 = np.linspace(-np.pi, np.pi, num_sample)
        q2 = np.linspace(-np.pi, np.pi, num_sample)
        q3 = np.linspace(-np.pi, np.pi, num_sample)
        q4 = np.linspace(-np.pi, np.pi, num_sample)
        q5 = np.linspace(-np.pi, np.pi, num_sample)
        q6 = np.linspace(-np.pi, np.pi, num_sample)
        X, Y, Z, N, M, O = np.meshgrid(q1, q2, q3, q4, q5, q6)
        joints = np.column_stack(
            [
                X.ravel(),
                Y.ravel(),
                Z.ravel(),
                N.ravel(),
                M.ravel(),
                O.ravel(),
            ]
        )
        jointstat = np.zeros((joints.shape[0],))
        for i in tqdm.tqdm(range(joints.shape[0])):
            iscollision = self.collision_check_at_config(joints[i])
            if iscollision:
                jointstat[i] = 1  # collision
            else:
                jointstat[i] = -1  # no collision
        dataset = np.column_stack([joints, jointstat])
        dataset = np.array(dataset)
        np.save(
            os.path.join(Constants.rsrcpath, "ur5e_collision_dataset.npy"), dataset
        )


class OMPLPlanner:

    def __init__(self, collision_checker):
        self.collision_checker = collision_checker

        self.space = ob.RealVectorStateSpace(6)
        self.bounds = ob.RealVectorBounds(6)
        self.limit6 = [
            2 * np.pi,
            2 * np.pi,
            np.pi,
            2 * np.pi,
            2 * np.pi,
            2 * np.pi,
        ]
        for i in range(6):
            self.bounds.setLow(i, -self.limit6[i])
            self.bounds.setHigh(i, self.limit6[i])
        self.bounds.setLow(1, -np.pi)
        self.bounds.setHigh(1, 0)
        self.space.setBounds(self.bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )
        # self.planner = og.BITstar(self.ss.getSpaceInformation())
        self.planner = og.ABITstar(self.ss.getSpaceInformation())
        # self.planner = og.AITstar(self.ss.getSpaceInformation())
        # self.planner.setRange(0.1)
        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        q = [state[0], state[1], state[2], state[3], state[4], state[5]]
        col = self.collision_checker(q)
        return not col

    def query_planning(self, start_list, goal_list):
        # Important!
        # Clear previous planning data to ensure fresh planning because caching
        self.ss.clear()

        start = ob.State(self.space)
        start[0] = start_list[0]
        start[1] = start_list[1]
        start[2] = start_list[2]
        start[3] = start_list[3]
        start[4] = start_list[4]
        start[5] = start_list[5]
        goal = ob.State(self.space)
        goal[0] = goal_list[0]
        goal[1] = goal_list[1]
        goal[2] = goal_list[2]
        goal[3] = goal_list[3]
        goal[4] = goal_list[4]
        goal[5] = goal_list[5]

        dist = np.linalg.norm(np.array(goal_list) - np.array(start_list))

        self.ss.setStartAndGoalStates(start, goal)
        status = self.ss.solve(100.0)
        print("Plan from ", start_list, " to ", goal_list, "estimate cost:", dist)
        (
            print("EXACT")
            if status.getStatus() == status.EXACT_SOLUTION
            else print("Invalid result")
        )
        if status.getStatus() == status.EXACT_SOLUTION:
            self.ss.simplifySolution()
            path = self.ss.getSolutionPath()
            path_cost = path.length()

            print("Found solution:")
            print(f"Path cost: {path_cost}")
            print(self.ss.getSolutionPath())

            pathlist = []
            for i in range(path.getStateCount()):
                pi = path.getState(i)
                pathlist.append([pi[0], pi[1], pi[2], pi[3], pi[4], pi[5]])
            return pathlist, path_cost
        else:
            print("No solution found")
            return None


def example_planning():
    robot = UR5eBullet("no_gui")
    model_id = robot.load_models_other(Constants.model_list_shelf)
    # 4.439302803306453
    planner = OMPLPlanner(robot.collision_check_at_config)
    qs = np.array([5.177, -0.957, 0.726, 0.231, 2.036, -3.142])
    qg = np.array([5.177, -0.56, 1.384, 2.318, 4.248, -6.283])
    result = planner.query_planning(qs, qg)
    if result is not None:
        pathlist, path_cost = result
        print("Path list:")
        print(pathlist)
        print(f"Path cost: {path_cost}")


def generate_linear_tasks_transformation(
    s=[1, 1, 1],
    e=[1, -1, 1],
    quat=[0.0, 0.707106, 0.0, 0.707106],
    num_tasks=10,
):
    t = np.linspace(s, e, num_tasks)
    Hlist = [np.eye(4) for _ in range(num_tasks)]
    for i in range(num_tasks):
        Hlist[i][:3, 3] = t[i]
        Hlist[i][:3, :3] = R.from_quat(quat).as_matrix()
    return Hlist


def generate_linear_dual_side_tasks_transformation():
    size = 4
    H1 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.6], [0.5, -0.5, 0.6], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H2 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.4], [0.5, -0.5, 0.4], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H3 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.2], [0.5, -0.5, 0.2], [0.0, 0.707106, 0.0, 0.707106], size
    )
    H4 = generate_linear_tasks_transformation(
        [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.707106, 0.0, 0.707106], size
    )
    Hlist = H1 + H2 + H3 + H4
    H1list = []
    for h in Hlist:
        H1list.append(convert_urdf_to_dh_frame(h))
    return Hlist + H1list


def simple_visualize():
    robot = UR5eBullet("gui")
    robot.load_slider()

    robot.set_visualizer_camera(*Constants.camera_exp3)

    try:
        while True:
            _, jp = robot.read_slider()
            robot.reset_array_joint_state(jp)
            p.stepSimulation()

    except KeyboardInterrupt:
        p.disconnect()


def collision_check():
    robot = UR5eBullet("gui")
    # model_id = robot.load_models_other(Constants.model_list)
    model_id = robot.load_models_other(Constants.model_list_shelf)

    robot.set_visualizer_camera(*Constants.camera_exp3)
    robot.draw_frame([1, 1, 1], [0, 0, 0, 1])

    try:
        while True:
            qKey = ord("c")
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                break
            # iscollide = robot.collsion_check(model_id[0])
            iscollide = robot.collision_check_all()
            # iscollide = robot.collision_check_at_config(
            #     [0.0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0]
            # )
            print(f"Collision: {iscollide}")
            p.stepSimulation()

    except KeyboardInterrupt:
        p.disconnect()


def kinematic_test():
    robot = RobotUR5eKin()

    q = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])
    H = robot.solve_fk(q)
    print("FK H:\n", H)

    M = robot.solve_manipulability(q)
    print("manipulability:", M)

    Jac = robot.solve_jacobian(q)
    print("Jacobian:\n", Jac)


if __name__ == "__main__":
    simple_visualize()
    # kinematic_test()
