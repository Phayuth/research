def allnode_RGG_sparse():
    points = np.random.uniform(-np.pi, np.pi, size=(200, 2))
    points_sparse = sparsify_nodes(points, eps=0.05 * 2 * np.pi)  # node sparse

    k = 5
    graph, kdt = build_graph(points, k)
    graph_sparse, kdt_sparse = build_graph(points_sparse, k)
    graph_sparse = prune_edges_triangle(graph_sparse, points_sparse, delta=0.1)

    rootnode = points_sparse[0]
    print(f"==>> rootnode: \n{rootnode}")
    goalnode = points_sparse[1]
    print(f"==>> goalnode: \n{goalnode}")
    gg, ggi = kdt_sparse.query(goalnode, k=10)
    ggiq = points_sparse[ggi]
    ch = ConvexHull(ggiq)
    chvq = ggiq[ch.vertices]

    fig, ax = plt.subplots()
    ax.plot(cspace_obs[:, 0], cspace_obs[:, 1], "ro", markersize=3)
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=100,
        c="lightgray",
        marker="x",
        label="Original Nodes",
    )
    ax.scatter(
        points_sparse[:, 0], points_sparse[:, 1], s=50, c="b", label="Sparse Nodes"
    )
    ax.scatter(
        rootnode[0], rootnode[1], s=100, c="r", marker="s", label="Root Node"
    )
    ax.scatter(
        goalnode[0], goalnode[1], s=100, c="g", marker="^", label="Goal Node"
    )

    cluster_polygon = plt.Polygon(
        chvq,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        label="Convex Hull of Neighbors",
    )
    ax.add_patch(cluster_polygon)

    for i, neighbors in graph.items():
        for j, _ in neighbors:
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                "r--",
                alpha=0.1,
            )

    for i, neighbors in graph_sparse.items():
        for j, _ in neighbors:
            ax.plot(
                [points_sparse[i, 0], points_sparse[j, 0]],
                [points_sparse[i, 1], points_sparse[j, 1]],
                "k-",
                alpha=0.4,
            )

    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid()
    plt.show()


def allnode_RGG_sparse_3d():
    # points = np.random.rand(100, 3)
    points = np.random.uniform(-np.pi, np.pi, size=(200, 3))

    # --- 1. node sparsification ---
    points = sparsify_nodes(points, eps=0.05 * 2 * np.pi)

    k = 5
    graph, kdt = build_graph(points, k)

    # --- 2. edge pruning ---
    graph = prune_edges_triangle(graph, points, delta=0.1)

    rootid = 0
    goalid = 1
    rootnode = points[rootid]
    goalnode = points[goalid]

    scene = trimesh.Scene()
    axis = trimesh.creation.axis(origin_size=0.05, axis_length=np.pi)
    box = trimesh.creation.box(extents=(2 * np.pi, 2 * np.pi, 2 * np.pi))
    box.visual.face_colors = [100, 150, 255, 40]
    scene.add_geometry(box)
    scene.add_geometry(axis)

    for i, neighbors in graph.items():
        for j, _ in neighbors:
            line = trimesh.load_path(
                np.array([points[i], points[j]]), color=[0, 0, 0, 40]
            )
            scene.add_geometry(line)
    scene.show()
