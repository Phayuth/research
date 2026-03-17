import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as SPolygon
from scipy.spatial import HalfspaceIntersection, ConvexHull


toppolygon = SPolygon([(-1.43, 5.97), (5.9, 1.84), (-4.74, 1.1)])
rightpolygon = SPolygon([(2.13, 0.98), (4.77, 3.89), (11.32, -0.27), (7.35, -2.47)])
bottompolygon = SPolygon([(-4.84, -1.42), (6.85, -0.03), (7.77, -5.75), (-1.35, -8.19)])
leftpolygon = SPolygon([(-5.47, 3.6), (-1.76, -0.77), (-5.62, -3.7)])
polys = [toppolygon, rightpolygon, bottompolygon, leftpolygon]


graph = np.array(
    [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ]
)

h1 = ConvexHull(toppolygon.exterior.coords)
h1eq = h1.equations
print(f"> h1eq: {h1eq}")


def show_gcs(graph, polys):
    centroids = np.array([[p.centroid.x, p.centroid.y] for p in polys])

    fig, ax = plt.subplots()
    # Draw the polygons
    for i, p in enumerate(polys):
        ax.add_patch(Polygon(list(p.exterior.coords), closed=True, fill=True, alpha=0.5))
    # Draw the edges of the polygons
    for i in range(len(polys)):
        for j in range(len(polys)):
            if graph[i, j] == 1:
                ax.quiver(
                    centroids[i, 0],
                    centroids[i, 1],
                    centroids[j, 0] - centroids[i, 0],
                    centroids[j, 1] - centroids[i, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="black",
                )
    # Draw the centroids
    ax.plot(centroids[:, 0], centroids[:, 1], "o", color="black", alpha=0.5)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)
    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 15)
    ax.set_aspect("equal")
    ax.set_title("Polygonal Regions")
    plt.show()

show_gcs(graph, polys)