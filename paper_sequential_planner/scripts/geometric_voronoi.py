import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, Delaunay
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as shapelyPolygon
import time

np.set_printoptions(linewidth=1000, suppress=True, precision=2)
np.random.seed(42)


def vrn2d():
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    points = np.meshgrid(x, y)
    points = np.array([points[0].ravel(), points[1].ravel()]).T

    # points = np.array(
    #     [
    #         [0, 0],
    #         [0, 1],
    #         [0, 2],
    #         [1, 0],
    #         [1, 1],
    #         [1, 2],
    #         [2, 0],
    #         [2, 1],
    #         [2, 2],
    #     ]
    # )
    vor = Voronoi(points)

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.show()


def vrn6d():
    rng = np.random.default_rng()
    points = rng.random((1000, 6))  # 10 s to make
    points = rng.random((10000, 6))  # 255 s to make

    t0 = time.time()
    vor = Voronoi(points)
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.2f} s")


def __test_voronoi():
    rng = np.random.default_rng(9)
    points = rng.random((10, 2))
    print(f"> points: {points}")

    vor = Voronoi(points)
    print(f"> vor: {vor}")
    print(f"> vor.points: {vor.points}")
    print(f"> vor.vertices: {vor.vertices}")
    print(f"> vor.regions: {vor.regions}")

    # fig, ax = plt.subplots()
    # voronoi_plot_2d(vor, ax)
    # for i, p in enumerate(points):
    #     ax.text(p[0], p[1], f"p:{i}:{p}")
    # for i, v in enumerate(vor.vertices):
    #     ax.text(v[0], v[1], f"v:{i}:{v}")
    # for k, r in enumerate(vor.regions):
    #     if len(r) == 0:
    #         continue
    #     if -1 in r:
    #         continue
    #     vv = [vor.vertices[j] for j in r]
    #     vv = np.concatenate(vv)
    #     poly = Polygon([vor.vertices[j] for j in r], facecolor="none", edgecolor="r")
    #     ax.add_patch(poly)

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_aspect("equal")
    # plt.show()

    p1 = shapelyPolygon(vor.vertices[vor.regions[4]])
    p2 = shapelyPolygon(vor.vertices[vor.regions[6]])
    p3 = shapelyPolygon(vor.vertices[vor.regions[7]])

    d = p1.distance(p2)
    ii = p1.intersection(p2)
    id = p1.intersects(p2)

    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax)
    for i, p in enumerate(points):
        ax.text(p[0], p[1], f"p:{i}:{p}")
    for i, v in enumerate(vor.vertices):
        ax.text(v[0], v[1], f"v:{i}:{v}")
    p11 = Polygon(p1.exterior, facecolor="r", edgecolor="r")
    ax.add_patch(p11)
    p22 = Polygon(p2.exterior, facecolor="b", edgecolor="b")
    ax.add_patch(p22)
    p33 = Polygon(p3.exterior, facecolor="g", edgecolor="g")
    ax.add_patch(p33)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.show()

    # vorrr = Voronoi(points, incremental=True)
    # print(f"> vorrr: {vorrr}")
    # vorrr.add_points(np.array([[0.5, 0.5]]))

    convx1 = ConvexHull(vor.vertices[vor.regions[4]])
    print(f"> convx1: {convx1}")
    eq1 = convx1.equations
    print(f"> eq1: {eq1}")

    fig, ax = plt.subplots()
    ax.plot(convx1.points[:, 0], convx1.points[:, 1], "o")
    p11 = Polygon(p1.exterior, facecolor="r", edgecolor="r")
    ax.add_patch(p11)
    for i in range(convx1.points.shape[0]):
        ax.axline(
            (convx1.points[i, 0], convx1.points[i, 1]),
            (
                convx1.points[(i + 1) % convx1.points.shape[0], 0],
                convx1.points[(i + 1) % convx1.points.shape[0], 1],
            ),
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    vrn2d()
    # vrn6d()
    # __test_voronoi()
