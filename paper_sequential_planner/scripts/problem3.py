import numpy as np
from spatial_geometry.spatial_shape import ShapeRectangle
import os
import shapely
from shapely.geometry import LineString, box
from shapely.ops import nearest_points

rsrc = os.environ["RSRC_DIR"] + "/rnd_torus/"


class Env:

    def __init__(self):
        pass

    def task_map(self):
        return [
            ShapeRectangle(x=-0.7, y=1.3, h=2, w=2.2),
            ShapeRectangle(x=2, y=-2.0, h=1, w=4.0),
            ShapeRectangle(x=-3, y=-3, h=1.25, w=2),
        ]


class PlanarRR:

    def __init__(self):
        # kinematic
        self.alpha1 = 0
        self.alpha2 = 0
        self.d1 = 0
        self.d2 = 0
        self.a1 = 2
        self.a2 = 2

    def forward_kinematic(self, theta):
        theta1 = theta[0, 0]
        theta2 = theta[1, 0]

        x = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
        y = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)

        link_end_pose = []
        link_end_pose.append([0, 0])

        # link 1 pose
        x1 = self.a1 * np.cos(theta1)
        y1 = self.a1 * np.sin(theta1)
        link_end_pose.append([x1, y1])

        # link 2 pose
        x2 = self.a1 * np.cos(theta1) + self.a2 * np.cos(theta1 + theta2)
        y2 = self.a1 * np.sin(theta1) + self.a2 * np.sin(theta1 + theta2)
        link_end_pose.append([x2, y2])

        return link_end_pose

    def distance_to_obstacles(self, theta, shapes):
        # get link endpoint coordinates (list of [x,y])
        link_points = self.forward_kinematic(theta)

        # build LineString objects for each link segment
        links = []
        for i in range(len(link_points) - 1):
            p1 = tuple(link_points[i])
            p2 = tuple(link_points[i + 1])
            links.append(LineString([p1, p2]))

        results = []
        for li, link in enumerate(links):
            for sj, shp in enumerate(shapes):
                # try to build a shapely polygon for the shape
                poly = None
                # try typical attributes first (center-based rectangle)
                cx = getattr(shp, "x", None)
                cy = getattr(shp, "y", None)
                h = getattr(shp, "h", None)
                w = getattr(shp, "w", None)
                if None not in (cx, cy, h, w):
                    # here x,y are the bottom-left corner of the rectangle
                    minx = float(cx)
                    miny = float(cy)
                    maxx = minx + float(w)
                    maxy = miny + float(h)
                    poly = box(minx, miny, maxx, maxy)
                else:
                    # fallback: if the shape can provide a shapely geometry
                    try:
                        poly = shp.to_shapely()
                    except Exception:
                        # as a last resort skip this shape
                        continue

                # compute distance and nearest points
                dist = link.distance(poly)
                p_link, p_poly = nearest_points(link, poly)
                results.append(
                    {
                        "link_idx": li,
                        "shape_idx": sj,
                        "distance": float(dist),
                        "link_point": (float(p_link.x), float(p_link.y)),
                        "shape_point": (float(p_poly.x), float(p_poly.y)),
                    }
                )

        if not results:
            return None, []

        best = min(results, key=lambda r: r["distance"])
        return best, results

    def plot_configuration(
        self, theta, shapes, best=None, results=None, ax=None, show=True
    ):
        """Plot the robot links, the rectangle obstacles, and (optionally) highlight the nearest points.

        This imports matplotlib lazily so the module doesn't require plotting dependencies unless called.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle as MplRect
        except Exception as e:
            raise RuntimeError("matplotlib is required for plotting: " + str(e))

        link_points = self.forward_kinematic(theta)

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

        # draw shapes
        for shp in shapes:
            cx = getattr(shp, "x", None)
            cy = getattr(shp, "y", None)
            h = getattr(shp, "h", None)
            w = getattr(shp, "w", None)
            if None not in (cx, cy, h, w):
                # x,y are bottom-left corner
                minx = float(cx)
                miny = float(cy)
                rect = MplRect(
                    (minx, miny),
                    float(w),
                    float(h),
                    facecolor="#d3d3d3",
                    edgecolor="k",
                    alpha=0.6,
                )
                ax.add_patch(rect)
            else:
                try:
                    poly = shp.to_shapely()
                    xs, ys = poly.exterior.xy
                    ax.fill(xs, ys, facecolor="#d3d3d3", edgecolor="k", alpha=0.6)
                except Exception:
                    pass

        # draw links
        xs = [p[0] for p in link_points]
        ys = [p[1] for p in link_points]
        ax.plot(xs, ys, "-o", color="blue", label="robot links")

        # optionally highlight nearest points
        if best is not None:
            lp = best.get("link_point")
            sp = best.get("shape_point")
            if lp is not None and sp is not None:
                ax.plot([lp[0]], [lp[1]], "ro", label="link nearest")
                ax.plot([sp[0]], [sp[1]], "go", label="shape nearest")
                ax.plot(
                    [lp[0], sp[0]], [lp[1], sp[1]], "r--", label="min distance"
                )
                ax.set_title(
                    f"Min dist {best['distance']:.3f} (link {best['link_idx']} -> shape {best['shape_idx']})"
                )

        ax.set_aspect("equal", "box")
        ax.grid(True)
        if created_fig and show:
            ax.legend()
            plt.show()
        return ax


if __name__ == "__main__":
    # simple demo run
    env = Env()
    shapes = env.task_map()
    robot = PlanarRR()
    # sample joint angles
    theta = np.array([[np.pi / 4.0], [np.pi / 4.0]])
    try:
        best, results = robot.distance_to_obstacles(theta, shapes)
        print("Best:", best)
        # also print all distances briefly
        print("All distances (link_idx, shape_idx, distance):")
        for r in results:
            print(r["link_idx"], r["shape_idx"], r["distance"])
        # plot configuration and distances
        try:
            robot.plot_configuration(theta, shapes, best=best, results=results)
        except RuntimeError as e:
            print("Plotting skipped:", e)
    except RuntimeError as e:
        print("RuntimeError:", e)
    except Exception as e:
        print("Error during demo run:", e)
