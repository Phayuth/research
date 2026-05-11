import os
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
from env_planarrr import PlanarRR, RobotScene
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
rsrc = os.environ["RSRC_DIR"]


class OMPLPlanner:

    def __init__(self, collision_checker, limit=np.pi):
        self.collision_checker = collision_checker
        self.dof = 2
        self.space = ob.RealVectorStateSpace(self.dof)
        self.bounds = ob.RealVectorBounds(self.dof)
        self.limit2 = [limit, limit]
        for i in range(self.dof):
            self.bounds.setLow(i, -self.limit2[i])
            self.bounds.setHigh(i, self.limit2[i])
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
        q = [state[0], state[1]]
        col = self.collision_checker(q)
        return not col

    def query_planning(self, start_list, goal_list):
        # Important!
        # Clear previous planning data to ensure fresh planning because caching
        self.ss.clear()

        start = ob.State(self.space)
        start[0] = start_list[0]
        start[1] = start_list[1]
        goal = ob.State(self.space)
        goal[0] = goal_list[0]
        goal[1] = goal_list[1]

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
                pathlist.append([pi[0], pi[1]])
            return pathlist, path_cost
        else:
            print("No solution found")
            return None


class OMPLRoadMapPlanner:

    def __init__(self, collision_checker, limit=np.pi):
        self.collision_checker = collision_checker
        self.dof = 2
        self.space = ob.RealVectorStateSpace(self.dof)
        self.bounds = ob.RealVectorBounds(self.dof)
        self.limit2 = [limit, limit]
        for i in range(self.dof):
            self.bounds.setLow(i, -self.limit2[i])
            self.bounds.setHigh(i, self.limit2[i])
        self.space.setBounds(self.bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.isStateValid)
        )

        self.planner = og.PRMstar(self.ss.getSpaceInformation())
        self.ss.setPlanner(self.planner)
        self.roadmap_data = None

    def isStateValid(self, state):
        q = [state[0], state[1]]
        col = self.collision_checker(q)
        return not col

    def construct_roadmap(self, time_limit=30.0):
        # Set planning time
        self.ss.setup()
        # self.planner.setNumSamples(num_samples)

        # Construct the roadmap
        print(f"Constructing roadmap for {time_limit} seconds...")
        self.planner.constructRoadmap(
            ob.timedPlannerTerminationCondition(time_limit)
        )
        data = ob.PlannerData(self.ss.getSpaceInformation())
        storage = ob.PlannerDataStorage()

        self.planner.getPlannerData(data)
        ne = data.numEdges()
        nv = data.numVertices()
        print(f"Roadmap constructed with {nv} vertices and {ne} edges.")

        # Save the roadmap to a file
        graphml_string = data.printGraphML()
        # with open("graph.graphml", "w") as f:
        #     f.write(graphml_string)

        return graphml_string


robot = PlanarRR()
scene = RobotScene(robot, None)
planner = OMPLPlanner(scene.collision_checker)
planner_roadmap = OMPLRoadMapPlanner(scene.collision_checker)
# graphml_string = planner_roadmap.construct_roadmap(time_limit=5.0)
# graphmlpath = "graph.graphml"
# graphml = nx.read_graphml(graphmlpath)
# graphml = nx.parse_graphml(graphml_string)

# qs = [0.0, 0.0]
# qg = [np.pi / 2, np.pi / 2]
# path, cost = planner.query_planning(qs, qg)
# cspace_obs = np.load(os.path.join(rsrc, "cspace_obstacles.npy"))
# fig, ax = plt.subplots()
# ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=1)
# ax.plot(xy[:, 0], xy[:, 1], "kx", markersize=3, label="Roadmap Vertices")
# ax.plot(qs[0], qs[1], "go", markersize=10, label="Start")
# ax.plot(qg[0], qg[1], "bo", markersize=10, label="Goal")
# if path is not None:
#     path = np.array(path)
#     ax.plot(path[:, 0], path[:, 1], "k-", linewidth=2, label="Planned Path")
# ax.legend()
# ax.set_aspect("equal", "box")
# ax.set_xlim(-np.pi, np.pi)
# ax.set_ylim(-np.pi, np.pi)
# ax.grid(True)
# plt.show()


# Benchmark Bulk Planning
n = 7000
Qs = np.random.rand(n, 2) * 2 * np.pi - np.pi
Qg = np.random.rand(n, 2) * 2 * np.pi - np.pi


d = 6  # dimension
nb = 100  # database size
nq = 10  # nb of queries
xb = np.random.random((nb, d)).astype("float32")
xb[:, 0] += np.arange(nb) / 1000.0
print(f"==>> xb: \n{xb}")

xq = np.random.random((nq, d)).astype("float32")
xq[:, 0] += np.arange(nq) / 1000.0
print(f"==>> xq: \n{xq}")

raise
import faiss
res = faiss.StandardGpuResources()  # use a single GPU

## Using a flat index
index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

# make it a flat GPU index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

gpu_index_flat.add(xb)  # add vectors to the index
print(gpu_index_flat.ntotal)

k = 4  # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
print(I[:5])  # neighbors of the 5 first queries
print(I[-5:])  # neighbors of the 5 last queries


## Using an IVF index
nlist = 100
quantizer = faiss.IndexFlatL2(d)  # the other index
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search

# make it an IVF GPU index
gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)

assert not gpu_index_ivf.is_trained
gpu_index_ivf.train(xb)  # add vectors to the index
assert gpu_index_ivf.is_trained

gpu_index_ivf.add(xb)  # add vectors to the index
print(gpu_index_ivf.ntotal)

k = 4  # we want to see 4 nearest neighbors
D, I = gpu_index_ivf.search(xq, k)  # actual search
print(I[:5])  # neighbors of the 5 first queries
print(I[-5:])  # neighbors of the 5 last queries
