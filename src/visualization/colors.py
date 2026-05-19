import numpy as np


def hex_to_rgb(hex_str: str) -> list[int]:
    h = hex_str.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]


def normalize_color_map(color_map: dict) -> dict[int, list[int]]:
    """Normalize color_map values to [r, g, b] int lists (0-255).

    Accepts both hex strings ('#1a2b3c') and [r, g, b] lists.
    """
    out = {}
    for k, v in color_map.items():
        if isinstance(v, str):
            out[int(k)] = hex_to_rgb(v)
        else:
            out[int(k)] = [int(c) for c in v]
    return out


def labels_to_colors(labels: np.ndarray, color_map: dict) -> np.ndarray:
    """Map label array to (N, 3) float32 RGB array in [0, 1].

    Unknown labels are mapped to gray.
    """
    norm = normalize_color_map(color_map)
    default = [128, 128, 128]
    colors = np.array(
        [norm.get(int(l), default) for l in labels], dtype=np.float32
    )
    return colors / 255.0


def auto_color_map(num_classes: int) -> dict[int, list[int]]:
    """Generate a distinct color map when no color config is available."""
    cmap = _get_cmap("tab20", num_classes)
    return {i: [int(c * 255) for c in cmap(i % 20)[:3]] for i in range(num_classes)}


def _get_cmap(name: str, n: int):
    import matplotlib.pyplot as plt
    return plt.get_cmap(name)
