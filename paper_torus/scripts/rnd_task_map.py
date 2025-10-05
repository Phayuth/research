import numpy as np
from spatial_geometry.spatial_shape import ShapeRectangle


class PaperTorusIFAC2025:

    def __init__(self):
        self.xlim = (-3, 5)
        self.ylim = (-3, 3)
        self.thetas = np.array([[2.68, -0.70], [-2.85, 1.73]])
        self.qstart_text = [1.3, -1, "$q_s$"]
        self.qgoal_text = [-0.9, 0.5, "$q_g$"]

    def task_map(self):
        return [
            ShapeRectangle(x=-0.7, y=1.3, h=2, w=2.2),
            ShapeRectangle(x=2, y=-2.0, h=1, w=4.0),
            ShapeRectangle(x=-3, y=-3, h=1.25, w=2),
        ]


class PaperX202X:

    def __init__(self):
        self.xlim = (-3, 3)
        self.ylim = (-3, 3)
        self.thetas = None

    def task_map(self):
        return [
            ShapeRectangle(x=-0.7, y=1.3, h=2, w=2.2),
            ShapeRectangle(x=2, y=-2.0, h=1, w=4.0),
            ShapeRectangle(x=-3, y=-3, h=1.25, w=2),
        ]
