import numpy as np
from spatial_geometry.spatial_shape import ShapeRectangle


class PaperICCAS2024:

    def __init__(self):
        self.xlim = (-3, 5)
        self.ylim = (-3, 3)
        self.thetas = None

    def task_map(self):
        return [
            ShapeRectangle(x=2, y=2, h=2, w=2),
            ShapeRectangle(x=-4, y=2, h=2, w=2),
            ShapeRectangle(x=2, y=-4, h=2, w=2),
            ShapeRectangle(x=-4, y=-4, h=2, w=2),
        ]
