"""
SVG Optimization Guide for Matplotlib Plots with Dense Data

This file demonstrates various strategies to control which parts of your plot
are rendered as images (rasterized) vs vectors in SVG format for optimal
file size and rendering performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure matplotlib for optimal SVG rendering
plt.rcParams['svg.fonttype'] = 'none'  # Render text as text, not paths
plt.rcParams['figure.dpi'] = 150  # Higher DPI for rasterized elements


def demonstrate_svg_optimization():
    """Demonstrate different strategies for SVG optimization"""

    # Generate sample data
    np.random.seed(42)

    # Dense data (many points) - good candidate for rasterization
    dense_points_x = np.random.normal(0, 1, 10000)
    dense_points_y = np.random.normal(0, 1, 10000)

    # Sparse important data - keep as vectors
    important_points_x = np.array([0, 1, -1, 0.5, -0.5])
    important_points_y = np.array([0, 1, -1, -0.5, 0.5])

    # Path data - keep as vectors for crisp lines
    path_x = np.linspace(-2, 2, 50)
    path_y = np.sin(path_x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # =============== BAD EXAMPLE (left plot) ===============
    ax1.set_title("BAD: All elements as vectors\n(Large SVG file)")

    # Dense scatter plot as vectors (BAD for SVG)
    ax1.scatter(dense_points_x, dense_points_y, s=1, alpha=0.3,
               color='gray', label='Dense data (vectors)')

    # Important points as vectors (OK)
    ax1.scatter(important_points_x, important_points_y, s=100,
               color='red', edgecolor='black', linewidth=2,
               label='Important points', zorder=10)

    # Path as vectors (OK)
    ax1.plot(path_x, path_y, 'b-', linewidth=3, label='Path')

    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # =============== GOOD EXAMPLE (right plot) ===============
    ax2.set_title("GOOD: Selective rasterization\n(Optimized SVG)")

    # Dense scatter plot as raster (GOOD for SVG)
    ax2.scatter(dense_points_x, dense_points_y, s=1, alpha=0.3,
               color='gray', label='Dense data (rasterized)',
               rasterized=True)  # KEY: rasterize dense data

    # Important points as vectors (GOOD - keep crisp)
    ax2.scatter(important_points_x, important_points_y, s=100,
               color='red', edgecolor='black', linewidth=2,
               label='Important points', zorder=10)

    # Path as vectors (GOOD - keep crisp lines)
    ax2.plot(path_x, path_y, 'b-', linewidth=3, label='Path')

    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('svg_optimization_demo.svg', format='svg', bbox_inches='tight')
    plt.show()


def strategy_examples():
    """Show specific strategies for different plot elements"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Strategy 1: Rasterize dense scatter plots
    np.random.seed(42)
    x = np.random.normal(0, 1, 5000)
    y = np.random.normal(0, 1, 5000)

    ax1.scatter(x, y, s=1, alpha=0.5, rasterized=True)
    ax1.set_title("Strategy 1: Dense scatter\n(rasterized=True)")

    # Strategy 2: Rasterize complex patches/polygons
    for i in range(100):
        circle = patches.Circle((np.random.uniform(-2, 2), np.random.uniform(-2, 2)),
                               0.1, facecolor=np.random.rand(3), alpha=0.7)
        circle.set_rasterized(True)  # Rasterize individual patches
        ax2.add_patch(circle)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_title("Strategy 2: Many patches\n(set_rasterized(True))")

    # Strategy 3: Keep important elements as vectors
    ax3.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)),
             'b-', linewidth=2, label='Main curve')
    ax3.scatter([2, 4, 6, 8], [np.sin(2), np.sin(4), np.sin(6), np.sin(8)],
               s=100, color='red', zorder=10, label='Key points')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title("Strategy 3: Important elements\n(keep as vectors)")

    # Strategy 4: Mixed approach for network graphs
    # Many edges (rasterize) + important nodes (vectors)
    angles = np.linspace(0, 2*np.pi, 50, endpoint=False)
    x_circle = np.cos(angles)
    y_circle = np.sin(angles)

    # Many edges (rasterized)
    for i in range(len(angles)):
        for j in range(i+1, len(angles)):
            if np.random.random() < 0.1:  # Random connections
                ax4.plot([x_circle[i], x_circle[j]], [y_circle[i], y_circle[j]],
                        'gray', alpha=0.3, linewidth=0.5, rasterized=True)

    # Important nodes (vectors)
    ax4.scatter(x_circle, y_circle, s=50, color='blue', edgecolor='black',
               linewidth=1, zorder=10)
    ax4.set_title("Strategy 4: Network graph\n(edges rasterized, nodes vector)")
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('svg_strategies.svg', format='svg', bbox_inches='tight')
    plt.show()


def advanced_configuration():
    """Advanced SVG optimization configurations"""

    # Set global rasterization DPI
    plt.rcParams['savefig.dpi'] = 300  # High DPI for rasterized elements

    # Example with your specific use case
    fig, ax = plt.subplots(figsize=(8, 6))

    # Simulate your collision points data
    collision_points = np.random.uniform(-5, 5, (10000, 2))

    # Simulate tree/graph edges
    num_edges = 1000
    for i in range(num_edges):
        start = np.random.uniform(-5, 5, 2)
        end = start + np.random.normal(0, 0.5, 2)
        ax.plot([start[0], end[0]], [start[1], end[1]],
               'gray', alpha=0.3, linewidth=0.5, rasterized=True)

    # Dense collision points (rasterized)
    ax.scatter(collision_points[:, 0], collision_points[:, 1],
              s=1, color='darkcyan', alpha=0.7, rasterized=True)

    # Important path points (vectors - stay crisp)
    path_points = np.array([[-4, -4], [-2, 0], [0, 2], [2, 0], [4, 4]])
    ax.plot(path_points[:, 0], path_points[:, 1],
           'blue', linewidth=3, marker='o', markersize=8,
           markerfacecolor='yellow', markeredgecolor='black')

    # Start/goal points (vectors - important to stay crisp)
    ax.scatter([-4], [-4], s=200, color='green',
              edgecolor='black', linewidth=2, marker='s', zorder=20)
    ax.scatter([4], [4], s=200, color='red',
              edgecolor='black', linewidth=2, marker='s', zorder=20)

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True, alpha=0.3)
    ax.set_title("Optimized Plot for Your Use Case")

    # Save with optimized settings
    plt.savefig('optimized_robotics_plot.svg', format='svg',
               bbox_inches='tight', dpi=300)
    plt.show()


def quick_optimization_checklist():
    """Print a quick checklist for SVG optimization"""

    checklist = """
    ═══════════════════════════════════════════════════════════════
                        SVG OPTIMIZATION CHECKLIST
    ═══════════════════════════════════════════════════════════════

    ✅ RASTERIZE (render as image):
       • Dense scatter plots with >1000 points
       • Collision point clouds
       • Complex tree/graph structures with many edges
       • Filled contours or heatmaps
       • Any plot element with >500 individual objects

    ✅ KEEP AS VECTORS (crisp and scalable):
       • Text labels and axis labels
       • Important path lines
       • Start/goal markers
       • Legend elements
       • Grid lines
       • Mathematical symbols and equations

    ✅ CONFIGURATION SETTINGS:
       plt.rcParams['svg.fonttype'] = 'none'  # Text as text, not paths
       plt.rcParams['savefig.dpi'] = 300      # High DPI for rasterized parts

    ✅ USAGE EXAMPLES:
       # For scatter plots:
       ax.scatter(x, y, rasterized=True)

       # For line plots:
       ax.plot(x, y, rasterized=True)

       # For patches:
       patch.set_rasterized(True)

       # For collections:
       collection.set_rasterized(True)

    ✅ BENEFITS:
       • Smaller SVG file sizes
       • Faster rendering in browsers/viewers
       • Better performance in vector graphics software
       • Maintained quality for important elements

    ═══════════════════════════════════════════════════════════════
    """
    print(checklist)


if __name__ == "__main__":
    quick_optimization_checklist()
    demonstrate_svg_optimization()
    strategy_examples()
    advanced_configuration()