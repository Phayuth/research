import os
import numpy as np
import tqdm
import trimesh

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    ou.RNG.setSeed(42)

except ImportError:
    print("OMPL not available, limitted functionality without OMPL.")

np.random.seed(42)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class Spatial3R:

    def __init__(self):
        self.links = [1, 1]

    def fk_link(self, q):
        # base
        p0 = np.array([0, 0, 0])

        # first planar joint in local plane
        r1 = self.links[0] * np.cos(q[1])
        z1 = self.links[0] * np.sin(q[1])
        p1 = np.array([r1 * np.cos(q[0]), r1 * np.sin(q[0]), z1])

        # end-effector
        r2 = self.links[0] * np.cos(q[1]) + self.links[1] * np.cos(q[1] + q[2])
        z2 = self.links[0] * np.sin(q[1]) + self.links[1] * np.sin(q[1] + q[2])
        p2 = np.array([r2 * np.cos(q[0]), r2 * np.sin(q[0]), z2])

        return np.array([p0, p1, p2])

    def ik(self, X):
        x, y, z = X[0], X[1], X[2]
        L1, L2 = self.links[0], self.links[1]

        # Step 1: Pan angle
        q0 = np.arctan2(y, x)

        # Step 2: Radial distance in xy-plane
        r = np.sqrt(x**2 + y**2)

        # Step 3: 2R planar IK in (r,z) plane
        d_sq = r**2 + z**2
        d = np.sqrt(d_sq)

        # Check reachability
        if d > (L1 + L2) or d < abs(L1 - L2):
            return None  # Target unreachable

        cos_q2 = (d_sq - L1**2 - L2**2) / (2 * L1 * L2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)  # Numerical stability

        sin_q2_positive = np.sqrt(1 - cos_q2**2)

        # Elbow-up solution
        q2_up = np.arctan2(sin_q2_positive, cos_q2)
        q1_up = np.arctan2(z, r) - np.arctan2(
            L2 * sin_q2_positive, L1 + L2 * cos_q2
        )
        sol_up = np.array([q0, q1_up, q2_up])

        # Elbow-down solution
        q2_down = np.arctan2(-sin_q2_positive, cos_q2)
        q1_down = np.arctan2(z, r) - np.arctan2(
            L2 * (-sin_q2_positive), L1 + L2 * cos_q2
        )
        sol_down = np.array([q0, q1_down, q2_down])

        # Return 2x3 matrix: [elbow_up, elbow_down]
        return np.array([sol_up, sol_down])


class RobotScene:

    def __init__(self, robot, obstacles):
        self.robot = robot
        self.wspace_limits = [(-2, 2), (-2, 2), (0, 2)]
        self.boxsize = {
            "box1": ((0.5, 0.5, 1), (1.5, 0, 0)),
            "box2": ((0.5, 0.5, 1), (-1.5, 0, 0)),
            "box3": ((0.5, 0.5, 1), (0, 1.5, 0)),
            "box4": ((0.5, 0.5, 1), (0, -1.5, 0)),
            "box5": ((0.5, 0.5, 1), (0, 0, 1.5)),
            "box6": ((0.5, 0.5, 1), (0, 0, -1.5)),
        }
        self.obstacles = []
        for name, (size, pos) in self.boxsize.items():
            colbox = trimesh.creation.box(extents=size)
            colbox.apply_translation(pos)
            self.obstacles.append(colbox)
        self.arm_thickness = 0.05

        # collision manager for static obstacles
        self.collision_manager = trimesh.collision.CollisionManager()
        for k, obs in enumerate(self.obstacles):
            self.collision_manager.add_object(f"obs_{k}", obs)

    def _link_cylinder(self, p0, p1):
        v = p1 - p0
        h = np.linalg.norm(v)
        if h < 1e-9:
            return None
        cyl = trimesh.creation.cylinder(
            radius=self.arm_thickness, height=h, sections=32
        )
        T = trimesh.geometry.align_vectors([0, 0, 1], v / h)
        cyl.apply_transform(T)
        cyl.apply_translation((p0 + p1) * 0.5)
        return cyl

    def arm_collision(self, q):
        fk_links = self.robot.fk_link(q)
        cyl1 = self._link_cylinder(fk_links[0], fk_links[1])
        cyl2 = self._link_cylinder(fk_links[1], fk_links[2])
        return [c for c in [cyl1, cyl2] if c is not None]

    def distance_to_obstacles(self, q):
        # Approximate penetration check by sampling points on arm links.
        arm_cylinders = self.arm_collision(q)
        for i, cyl in enumerate(arm_cylinders):
            for j, obs in enumerate(self.obstacles):
                pts, _ = trimesh.sample.sample_surface(cyl, 3000)
                sdf = obs.nearest.signed_distance(pts)
                print(
                    f"link {i}, obs {j}: max_penetration={sdf.max():.6f}, min_signed={sdf.min():.6f}"
                )

    def collision_checker(self, q):
        for i, link_mesh in enumerate(self.arm_collision(q)):
            if self.collision_manager.in_collision_single(link_mesh):
                return True
        return False

    def cspace_dataset_collision(self):
        num_samples = 360
        q1 = np.linspace(-np.pi, np.pi, num_samples)
        q2 = np.linspace(-np.pi, np.pi, num_samples)
        q3 = np.linspace(-np.pi, np.pi, num_samples)
        Q1, Q2, Q3 = np.meshgrid(q1, q2, q3, indexing="ij")
        joints = np.column_stack([Q1.ravel(), Q2.ravel(), Q3.ravel()])
        print("Generated joint samples:", joints.shape)
        col = np.empty(joints.shape[0])
        for i in tqdm.tqdm(range(joints.shape[0])):
            q = joints[i]
            col[i] = self.collision_checker(q)
        joint_dataset = np.column_stack([joints, col])
        joint_dataset[:, 3] = np.where(
            joint_dataset[:, 3] == 0, -1, joint_dataset[:, 3]
        )
        np.save(os.path.join(rsrc, "spatial3r_cspace.npy"), joint_dataset)

    def show_env(self, q):
        # supports
        plane = trimesh.creation.box(extents=(5, 5, 0.01))
        plane.visual.face_colors = [200, 200, 200, 80]
        axis = trimesh.creation.axis(origin_size=0.05, axis_length=2.5)
        box = trimesh.creation.box(extents=(5, 5, 5))
        box.visual.face_colors = [100, 150, 255, 40]

        # arm links
        arm_cylinders = self.arm_collision(q)

        # scene setup
        scene = trimesh.Scene()
        scene.add_geometry(plane)
        scene.add_geometry(box)
        scene.add_geometry(axis)
        for cyl in arm_cylinders:
            scene.add_geometry(cyl)
        for colbox in self.obstacles:
            scene.add_geometry(colbox)
        scene.set_camera(angles=(0, 0, 0), distance=3.0, center=[0, 0, 0])
        scene.show()

    def _show_wsenv_debug(self, q):
        # supports
        plane = trimesh.creation.box(extents=(5, 5, 0.01))
        plane.visual.face_colors = [200, 200, 200, 80]
        axis = trimesh.creation.axis(origin_size=0.05, axis_length=2.5)
        box = trimesh.creation.box(extents=(5, 5, 5))
        box.visual.face_colors = [100, 150, 255, 40]

        # arm links
        arm_cylinders = self.arm_collision(q)

        # scene setup
        scene = trimesh.Scene()
        scene.add_geometry(plane)
        scene.add_geometry(box)
        scene.add_geometry(axis)
        for cyl in arm_cylinders:
            scene.add_geometry(cyl)
        for colbox in self.obstacles:
            scene.add_geometry(colbox)
        scene.set_camera(angles=(0, 0, 0), distance=3.0, center=[0, 0, 0])

        # add extras
        Xsample = sample_reachable_wspace(50)
        pc = trimesh.points.PointCloud(Xsample)
        scene.add_geometry(pc)
        scene.show()


class OMPLPlanner:

    def __init__(self, collision_checker):
        self.collision_checker = collision_checker
        self.dof = 3
        self.space = ob.RealVectorStateSpace(self.dof)
        self.bounds = ob.RealVectorBounds(self.dof)
        self.limit3 = [
            np.pi,
            np.pi,
            np.pi,
        ]
        for i in range(self.dof):
            self.bounds.setLow(i, -self.limit3[i])
            self.bounds.setHigh(i, self.limit3[i])
        self.space.setBounds(self.bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )
        self.planner = og.BITstar(self.ss.getSpaceInformation())
        # self.planner = og.ABITstar(self.ss.getSpaceInformation())
        # self.planner = og.AITstar(self.ss.getSpaceInformation())
        # self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        # self.planner.setRange(0.1)
        self.ss.setPlanner(self.planner)

    def isStateValid(self, state):
        q = np.array([state[0], state[1], state[2]])
        return not self.collision_checker(q)

    def query_planning(self, start_list, goal_list):
        # Important!
        # Clear previous planning data to ensure fresh planning because caching
        self.ss.clear()

        start = ob.State(self.space)
        start[0] = start_list[0]
        start[1] = start_list[1]
        start[2] = start_list[2]
        goal = ob.State(self.space)
        goal[0] = goal_list[0]
        goal[1] = goal_list[1]
        goal[2] = goal_list[2]

        lowest = np.linalg.norm(np.array(goal_list) - np.array(start_list))

        self.ss.setStartAndGoalStates(start, goal)
        status = self.ss.solve(5.0)
        print(
            "Plan from ", start_list, " to ", goal_list, "estimate cost:", lowest
        )
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
                pathlist.append([pi[0], pi[1], pi[2]])
            return pathlist, path_cost
        else:
            print("No solution found")
            return None


def sample_reachable_wspace(num_points):
    dof = 3
    u = np.random.normal(0.0, 1.0, (dof + 2, num_points))
    norms = np.linalg.norm(u, axis=0)
    u = u / norms
    scale = 2 * u[:3, :]  # The first N coordinates are uniform in a unit N ball
    return scale.T  # (ntasks, 3)


def wspace_ik(robot, Xtspace):
    """
    Xtaspace: (ntasks, 3)
    Compute AIK of all task points no matter whether they are reachable or not.
    NaN values will be used to indicate unreachable points.
    """
    ntasks = Xtspace.shape[0]
    num_sols = 2  # spatial3r has 2 IK solutions per task
    dof = 3
    Qaik = np.full((ntasks, num_sols, dof), np.nan)  # (ntasks, num_sols, dof)
    for taski in range(ntasks):
        q_sols = robot.ik(Xtspace[taski])
        if q_sols is not None:
            Qaik[taski] = q_sols
    return Qaik


def wspace_ik_validity(Qaik, robscene):
    """
    Compute the validity of each AIK solution in Qaik.
    1 = Valid
    -1 = NaN (no solution or unreachable)
    -2 = In collision
    -3 = awkward configuration (e.g. near singularity or joint limits)
    """
    ntasks = Qaik.shape[0]
    num_sols = Qaik.shape[1]
    limit = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
    eps = 1e-9
    MQaik_valid = np.full((ntasks, num_sols, 1), np.nan)  # (ntasks, num_sols, 1)
    for taski in range(Qaik.shape[0]):
        for solj in range(Qaik.shape[1]):
            q = Qaik[taski, solj]
            if np.isnan(q).any():
                MQaik_valid[taski, solj] = -1  # No solution
            else:
                col = robscene.collision_checker(q)
                if col:
                    MQaik_valid[taski, solj] = -2  # In collision
                else:
                    is_in_limit = np.all(
                        (q >= (limit[:, 0] - eps)) & (q <= (limit[:, 1] + eps))
                    )
                    if not is_in_limit:
                        MQaik_valid[taski, solj] = -3  # Awkward configuration
                    else:
                        MQaik_valid[taski, solj] = 1  # Valid
    return MQaik_valid.astype(int)


if __name__ == "__main__":
    from paper_sequential_planner.scripts.rtsp_solver import RTSP, GLKHHelper
    from paper_sequential_planner.scripts.rtsp_lazyprm import (
        separate_sample,
        build_graph,
        estimate_shortest_path,
    )

    robot = Spatial3R()
    scene = RobotScene(robot, None)
    planner = OMPLPlanner(scene.collision_checker)
    q = np.array([0.0, 0.0, np.pi])
    X = np.array([0.5, 0.5, 0.5])
    qik = robot.ik(X)
    q1 = qik[0]
    q2 = qik[1]
    # scene.collision_checker(q)
    # scene.distance_to_obstacles(q)
    # scene.cspace_dataset_collision()
    # scene.show_env(q)
    scene._show_wsenv_debug(q)

    ntasks = 30
    X = sample_reachable_wspace(ntasks)
    Qaik = wspace_ik(robot, X)
    Qaik_valid = wspace_ik_validity(Qaik, scene)
    print(f"==>> X: \n{X}")
    print(f"==>> Qaik: \n{Qaik}")
    print(f"==>> Qaik_valid: \n{Qaik_valid}")

    (
        task_reachable,
        num_qreachable,
        Q_reachable,
        cluster_ttc,
        cluster_ctt,
        taskspace_adjm,
        cspace_adjm,
    ) = RTSP.preprocess(X, Qaik, Qaik_valid)
    print(f"==>> task_reachable: \n{task_reachable}")
    print(f"==>> num_qreachable: \n{num_qreachable}")
    print(f"==>> Q_reachable: \n{Q_reachable}")
    print(f"==>> cluster_ttc: \n{cluster_ttc}")
    print(f"==>> cluster_ctt: \n{cluster_ctt}")
    print(f"==>> taskspace_adjm: \n{taskspace_adjm}")
    print(f"==>> cspace_adjm: \n{cspace_adjm}")

    num_unique_edges = RTSP.num_edges_unique(num_qreachable)
    print(f"==>> num_unique_edges: \n{num_unique_edges}")
    num_supercluster_edges = RTSP.num_supercluster_edges(ntasks)
    print(f"==>> num_supercluster_edges: \n{num_supercluster_edges}")

    # ------- Compute Initial Cost --------------------------------------
    cspace_adjm_euc_min = RTSP.edgecost_eucl_distance(Q_reachable)
    print(f"==>> cspace_adjm_euc_min: \n{cspace_adjm_euc_min}")
    cc = RTSP.get_cost_task_to_task(cluster_ttc, cspace_adjm_euc_min, 5, 7)
    print(f"==>> cc: \n{cc}")
    ast = np.argsort(cc)
    print(f"==>> ast: \n{ast}")
    # ------- End Compute Initial Cost ----------------------------------

    # ------ Estimation of Edges--------------------------------
    limit3 = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
    QfulRndfree, QfulRndcoll = separate_sample(
        collision_checker=scene.collision_checker,
        Qful_snum=5000,
        lmts=limit3,
    )
    graph, kdtree = build_graph(QfulRndfree, k=10, dist_thres=0.5)
    cspace_adjm, store_path, store_cost = RTSP.edgecost_colfree_distance(
        cspace_adjm,
        Q_reachable,
        estimate_shortest_path,
        {"Qfree": QfulRndfree, "graph": graph, "kdtree": kdtree},
    )
    taskspace_adjm = RTSP.update_taskspace_adjm(
        taskspace_adjm, cspace_adjm, cluster_ctt
    )
    print(f"==>> taskspace_adjm (updated with edge counts): \n{taskspace_adjm}")
    # ------ End Estimation of Edges-----------------------------

    s = np.array([-np.pi] * 3)
    g = np.array([np.pi] * 3)

    if True:
        # plot debug
        # dataset = np.load(os.path.join(rsrc, "spatial3r_cspace.npy"))
        # Qfree = dataset[dataset[:, -1] == 0][:, :3]
        # Qcoll = dataset[dataset[:, -1] == 1][:, :3]
        axis = trimesh.creation.axis(origin_size=0.05, axis_length=np.pi)
        box = trimesh.creation.box(extents=(2 * np.pi, 2 * np.pi, 2 * np.pi))
        box.visual.face_colors = [100, 150, 255, 40]

        # scene setup
        scene = trimesh.Scene()
        scene.add_geometry(box)
        scene.add_geometry(axis)

        Qr = trimesh.points.PointCloud(Q_reachable)
        # Qr = trimesh.points.PointCloud(Qcoll)
        scene.add_geometry(Qr)
        for i, p in enumerate(store_path.values()):
            # p = np.linspace(s, g, 10)  # n, 3
            edges = np.column_stack((np.arange(len(p) - 1), np.arange(1, len(p))))
            path = trimesh.load_path(p[edges])
            scene.add_geometry(path)

        scene.show()
