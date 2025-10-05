import numpy as np
from spatial_geometry.spatial_shape import ShapeRectangle


class PaperIJCAS2025:
    def __init__(self):
        self.xlim = (-4, 4)
        self.ylim = (-2, 4)

    def task_map(self):
        return [
            ShapeRectangle(x=-2.75, y=1, h=2, w=1),
            ShapeRectangle(x=1.5, y=2, h=2, w=1),
            ShapeRectangle(x=-0.75, y=-2.0, h=0.75, w=4.0),
        ]


if __name__ == "__main__":
    from simulator.sim_planar_rrr import RobotArm2DRRRSimulator
    from matplotlib import animation
    import matplotlib.pyplot as plt

    sim = RobotArm2DRRRSimulator(PaperIJCAS2025())

    # view
    thetas = np.array(
        [[0, 0, 0], [1, 1, np.pi / 3], [2, 2, np.pi / 2], [3, 3, np.pi]]
    ).T
    sim.plot_view(thetas)

    # play back
    path = np.linspace([0, 0, 0], [2 * np.pi, 0, 0], 100).T
    sim.play_back_path(path, animation)
