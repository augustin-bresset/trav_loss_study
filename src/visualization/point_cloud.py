"""2D projection views for 3D LiDAR point clouds."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .colors import labels_to_colors, normalize_color_map, auto_color_map


def _scatter2d(ax, x, y, colors, point_size, title, xlabel, ylabel):
    ax.scatter(x, y, c=colors, s=point_size, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")


def plot_bev(ax, pos: np.ndarray, colors: np.ndarray, point_size: float = 0.3):
    _scatter2d(ax, pos[:, 0], pos[:, 1], colors, point_size, "BEV (x-y)", "x (m)", "y (m)")


def plot_front_view(ax, pos: np.ndarray, colors: np.ndarray, point_size: float = 0.3):
    _scatter2d(ax, pos[:, 0], pos[:, 2], colors, point_size, "Front (x-z)", "x (m)", "z (m)")


def plot_side_view(ax, pos: np.ndarray, colors: np.ndarray, point_size: float = 0.3):
    _scatter2d(ax, pos[:, 1], pos[:, 2], colors, point_size, "Side (y-z)", "y (m)", "z (m)")


def _build_legend(ax, semantic_map: dict, color_map: dict):
    norm = normalize_color_map(color_map)
    patches = []
    for cls_id, cls_name in sorted(semantic_map.items()):
        rgb = norm.get(int(cls_id), [128, 128, 128])
        color = [c / 255.0 for c in rgb]
        patches.append(mpatches.Patch(color=color, label=f"{cls_id}: {cls_name}"))
    ax.legend(
        handles=patches,
        loc="upper left",
        fontsize=7,
        framealpha=0.8,
        ncol=max(1, len(patches) // 20),
    )
    ax.axis("off")


def plot_labeled_cloud(
    pos: np.ndarray,
    labels: np.ndarray,
    label_cfg: dict | None = None,
    title: str = "",
    point_size: float = 0.3,
    max_points: int = 80_000,
    save_path: str | None = None,
    show: bool = True,
):
    """Render BEV + front + side views with a label legend.

    Args:
        pos: (N, 3) float array of 3D coordinates.
        labels: (N,) int array of semantic labels.
        label_cfg: dict with keys 'semantic_map', 'color_map'. If None,
                   colors are generated automatically.
        title: figure suptitle.
        point_size: matplotlib scatter marker size.
        max_points: subsample if the cloud exceeds this size (speed).
        save_path: if set, save the figure to this path.
        show: call plt.show() at the end.
    """
    if len(pos) > max_points:
        idx = np.random.choice(len(pos), max_points, replace=False)
        pos, labels = pos[idx], labels[idx]

    if label_cfg is not None:
        color_map = label_cfg["color_map"]
        semantic_map = label_cfg.get("semantic_map", {})
        colors = labels_to_colors(labels, color_map)
    else:
        n_classes = int(labels.max()) + 1
        color_map = auto_color_map(n_classes)
        semantic_map = {i: str(i) for i in range(n_classes)}
        colors = labels_to_colors(labels, color_map)

    fig = plt.figure(figsize=(18, 6), facecolor="#111111")
    if title:
        fig.suptitle(title, color="white", fontsize=12)

    ax_bev = fig.add_subplot(1, 4, 1)
    ax_front = fig.add_subplot(1, 4, 2)
    ax_side = fig.add_subplot(1, 4, 3)
    ax_legend = fig.add_subplot(1, 4, 4)

    plot_bev(ax_bev, pos, colors, point_size)
    plot_front_view(ax_front, pos, colors, point_size)
    plot_side_view(ax_side, pos, colors, point_size)
    _build_legend(ax_legend, semantic_map, color_map)

    for ax in (ax_bev, ax_front, ax_side):
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    else:
        return fig
