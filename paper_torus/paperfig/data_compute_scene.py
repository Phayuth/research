import numpy as np
import pybullet as p
import pybullet_data
import os


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
        # (table_urdf, [0, 0, 0], [0, 0, 0, 1]),
        (pole_urdf, [0.3, 0.3, 0], [0, 0, 0, 1]),
        # (pole_urdf, [-0.3, 0.3, 0], [0, 0, 0, 1]),
        # (pole_urdf, [-0.3, -0.3, 0], [0, 0, 0, 1]),
        (pole_urdf, [0.3, -0.3, 0], [0, 0, 0, 1]),
    ]

    model_list_shelf = [
        (shelf_urdf, [0, 0.75, 0], [0, 0, 1, 0]),
        (shelf_urdf, [0, -0.75, 0], [0, 0, 0, 1]),
        (shelf_urdf, [-0.75, 0, 0], [0, 0, -0.5, 0.5]),
    ]

    model_box_strong_obstacle = [
        (pole_urdf, [0.3, 0, 0], [0, 0, 0, 1]),
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

    def load_slider(self):
        self.redSlider = p.addUserDebugParameter("red", 0, 1, 1)
        self.greenSlider = p.addUserDebugParameter("green", 0, 1, 0)
        self.blueSlider = p.addUserDebugParameter("blue", 0, 1, 0)
        self.alphaSlider = p.addUserDebugParameter("alpha", 0, 1, 0.5)

        self.j1s = p.addUserDebugParameter("joint_1", -2 * np.pi, 2 * np.pi, 0)
        self.j2s = p.addUserDebugParameter("joint_2", -2 * np.pi, 2 * np.pi, 0)
        self.j3s = p.addUserDebugParameter("joint_3", -2 * np.pi, 2 * np.pi, 0)
        self.j4s = p.addUserDebugParameter("joint_4", -2 * np.pi, 2 * np.pi, 0)
        self.j5s = p.addUserDebugParameter("joint_5", -2 * np.pi, 2 * np.pi, 0)
        self.j6s = p.addUserDebugParameter("joint_6", -2 * np.pi, 2 * np.pi, 0)

    def read_slider(self):
        red = p.readUserDebugParameter(self.redSlider)
        green = p.readUserDebugParameter(self.greenSlider)
        blue = p.readUserDebugParameter(self.blueSlider)
        alpha = p.readUserDebugParameter(self.alphaSlider)
        return red, green, blue, alpha

    def read_joint_slider(self):
        j1 = p.readUserDebugParameter(self.j1s)
        j2 = p.readUserDebugParameter(self.j2s)
        j3 = p.readUserDebugParameter(self.j3s)
        j4 = p.readUserDebugParameter(self.j4s)
        j5 = p.readUserDebugParameter(self.j5s)
        j6 = p.readUserDebugParameter(self.j6s)
        return j1, j2, j3, j4, j5, j6

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

    def normalize(self, v):
        v = np.array(v)
        return v / (np.linalg.norm(v) + 1e-9)

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
            axis_world = self.normalize(axis_world)

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

    def inverse_kin(self, positions, quaternions, q0):
        joint_angles = p.calculateInverseKinematics(
            self.ur5eID,
            self.gripperlinkid,
            positions,
            quaternions,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            jointDamping=self.joint_damp,
            currentPositions=q0,
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


def collision_check():
    robot = UR5eBullet("gui")
    model_id = robot.load_models_other(Constants.model_list)
    # model_id = robot.load_models_other(Constants.model_list_shelf)
    # model_id = robot.load_models_other(Constants.model_box_strong_obstacle)
    # robot.load_models_ghost(color=[0, 1, 0, 0.1])  # green ghost model
    # robot.load_models_ghost(color=[1, 0, 0, 0.1])  # red ghost model
    robot.load_slider()

    qs = (
        0.39683294296264565,
        -0.9920818805694598,
        1.5873310565948522,
        -0.8928737640380842,
        1.8518865108490026,
        -3.141592741012566,
    )
    # qg = robot.inverse_kin(
    #     positions=[0.3, -0.3, 0.5],
    #     quaternions=p.getQuaternionFromEuler([0, np.pi, 0]),
    #     q0=qs,
    # )
    qg = (
        -3.6376335620880056,
        -1.7857475280761683,
        -1.7196087837219274,
        -2.1825799942016664,
        -1.9180250167846644,
        -0.2645549774169931,
    )

    print(f"qs: {qs}")
    print(f"qg: {qg}")

    # robot.reset_array_joint_state(qs)
    # robot.reset_array_joint_state_ghost(qs, robot.ghost_model[0])
    # robot.reset_array_joint_state_ghost(qg, robot.ghost_model[1])
    robot.set_visualizer_camera(*Constants.camera_exp3)
    # robot.draw_frame([1, 1, 1], [0, 0, 0, 1])

    try:
        while True:
            qKey = ord("c")
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                break
            q = robot.read_joint_slider()
            robot.control_array_motors(q)
            # iscollide = robot.collsion_check(model_id[0])
            # iscollide = robot.collision_check_all()
            # iscollide = robot.collision_check_at_config(
            #     [0.0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0]
            # )
            # print(f"Collision: {iscollide}")
            q = robot.get_array_joint_positions()
            print(f"Current joint: {q}")
            p.stepSimulation()

    except KeyboardInterrupt:
        p.disconnect()


def joint_trajectory_visualize_ghost():
    robot = UR5eBullet("gui")
    robot.load_models_ghost(color=[0, 1, 0, 0.1])  # green ghost model
    robot.load_models_ghost(color=[1, 0, 0, 0.1])  # red ghost model
    robot.load_models_other(Constants.model_box_strong_obstacle)

    # camera for exp3
    robot.set_visualizer_camera(*Constants.camera_exp3)

    # path exp 2
    # qs = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0]
    # qg1_short = [-1.12, -1.86, 1.87, 0.0, np.pi / 2, 0.0]
    # n_steps = 1000
    # path_short = np.linspace(qs, qg1_short, n_steps)
    # path = path_short

    import pandas as pd

    # df_path = pd.read_csv("data_so2_path.csv")
    i = 2
    # for i in range(32):
    df_path = pd.read_csv("./paths/data_euclidean_path_goal_" + str(i) + ".csv")
    path = df_path.to_numpy()
    n_steps = path.shape[0]

    # initialize ghost model joint state
    robot.reset_array_joint_state(path[0])
    robot.reset_array_joint_state_ghost(path[0], robot.ghost_model[0])
    robot.reset_array_joint_state_ghost(path[-1], robot.ghost_model[-1])

    try:
        j = 0
        while True:
            nkey = ord("n")
            keys = p.getKeyboardEvents()
            if nkey in keys and keys[nkey] & p.KEY_WAS_TRIGGERED:
                q = path[j % n_steps]
                robot.reset_array_joint_state(q)
                j += 1
                p.stepSimulation()

    except KeyboardInterrupt:
        p.disconnect()


def joint_trajectory_visualize_ghost_perpath():
    robot = UR5eBullet("gui")
    robot.load_models_ghost(color=[0, 1, 0, 0.1])  # green ghost model
    robot.load_models_ghost(color=[1, 0, 0, 0.1])  # red ghost model
    robot.load_models_other(Constants.model_box_strong_obstacle)

    # camera for exp3
    robot.set_visualizer_camera(*Constants.camera_exp3)

    # path exp 2
    # qs = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0]
    # qg1_short = [-1.12, -1.86, 1.87, 0.0, np.pi / 2, 0.0]
    # n_steps = 1000
    # path_short = np.linspace(qs, qg1_short, n_steps)
    # path = path_short

    import pandas as pd

    # df_path = pd.read_csv("data_so2_path.csv")
    i = 0
    # for i in range(32):
    df_path = pd.read_csv("./paths/data_euclidean_path_goal_" + str(i) + ".csv")
    path = df_path.to_numpy()
    n_steps = path.shape[0]

    # initialize ghost model joint state
    robot.reset_array_joint_state(path[0])
    robot.reset_array_joint_state_ghost(path[0], robot.ghost_model[0])
    robot.reset_array_joint_state_ghost(path[-1], robot.ghost_model[-1])

    try:
        j = 0
        w = 0
        while True:
            nkey = ord("n")
            mkey = ord("m")
            keys = p.getKeyboardEvents()
            if nkey in keys and keys[nkey] & p.KEY_WAS_TRIGGERED:
                q = path[j % n_steps]
                robot.reset_array_joint_state(q)
                j += 1
                p.stepSimulation()
            if mkey in keys and keys[mkey] & p.KEY_WAS_TRIGGERED:
                df_path = pd.read_csv(
                    "./paths/data_euclidean_path_goal_" + str(w) + ".csv"
                )
                path = df_path.to_numpy()
                n_steps = path.shape[0]

                # initialize ghost model joint state
                robot.reset_array_joint_state(path[0])
                robot.reset_array_joint_state_ghost(path[0], robot.ghost_model[0])
                robot.reset_array_joint_state_ghost(
                    path[-1], robot.ghost_model[-1]
                )
                w += 1
                print(f"Switched to path goal {w-1}")

    except KeyboardInterrupt:
        p.disconnect()


def view_exp1_aik():
    import pandas as pd

    robot = UR5eBullet("gui")
    robot.load_models_ghost(color=[0, 1, 0, 0.1])  # green ghost model
    robot.load_models_ghost(color=[1, 0, 0, 0.1])  # red ghost model
    robot.load_models_ghost(color=[0, 0, 1, 0.1])  # blue ghost model
    robot.load_models_ghost(color=[1, 1, 0, 0.1])  # yellow ghost model
    robot.load_models_ghost(color=[1, 0, 1, 0.1])  # purple ghost model
    robot.load_models_ghost(color=[0, 1, 1, 0.1])  # cyan ghost model
    robot.load_models_ghost(color=[1, 1, 1, 0.1])  # white ghost model
    robot.load_models_ghost(color=[0, 0, 0, 0.1])  # black ghost model
    robot.load_models_ghost(color=[0.5, 0.5, 0.5, 0.1])  # gray ghost model
    # robot.load_models_other(Constants.model_list)

    # camera for exp3
    robot.set_visualizer_camera(*Constants.camera_exp3)

    df_qs = pd.read_csv("./data_ur5e_qs_inpi.csv")
    qs = df_qs.to_numpy().reshape(-1)

    df_Qik = pd.read_csv("./data_ur5e_Qik.csv")
    Qik = df_Qik.to_numpy()

    # initialize ghost model joint state
    robot.reset_array_joint_state(qs)
    robot.reset_array_joint_state_ghost(qs, robot.ghost_model[0])
    robot.reset_array_joint_state_ghost(Qik[0], robot.ghost_model[1])
    robot.reset_array_joint_state_ghost(Qik[1], robot.ghost_model[2])
    robot.reset_array_joint_state_ghost(Qik[2], robot.ghost_model[3])
    robot.reset_array_joint_state_ghost(Qik[3], robot.ghost_model[4])
    robot.reset_array_joint_state_ghost(Qik[4], robot.ghost_model[5])
    robot.reset_array_joint_state_ghost(Qik[5], robot.ghost_model[6])
    robot.reset_array_joint_state_ghost(Qik[6], robot.ghost_model[7])
    robot.reset_array_joint_state_ghost(Qik[7], robot.ghost_model[8])

    try:
        j = 0
        w = 0
        while True:
            nkey = ord("n")
            mkey = ord("m")
            keys = p.getKeyboardEvents()
            if nkey in keys and keys[nkey] & p.KEY_WAS_TRIGGERED:
                q = path[j % n_steps]
                robot.reset_array_joint_state(q)
                j += 1
                p.stepSimulation()
            if mkey in keys and keys[mkey] & p.KEY_WAS_TRIGGERED:
                df_path = pd.read_csv(
                    "./paths_exp1/data_exp1_path_goal_" + str(w) + ".csv"
                )
                path = df_path.to_numpy()
                n_steps = path.shape[0]

                # initialize ghost model joint state
                robot.reset_array_joint_state(path[0])
                w += 1
                print(f"Switched to path goal {w-1}")

    except KeyboardInterrupt:
        p.disconnect()


if __name__ == "__main__":
    # collision_check()
    # joint_trajectory_visualize_ghost()
    # joint_trajectory_visualize_ghost_perpath()
    view_exp1_aik()
