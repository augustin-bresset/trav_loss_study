"""Interactive 3-D LiDAR dataset viewer built on Open3D GUI.

Usage (programmatic):
    DatasetViewer.launch(dataset, label_cfg, start_idx=0)

Usage (CLI):
    python -m src.visualization --dataset goose --root /path/to/goose \\
        --cfg resources/goose_cfg.yaml --split train --idx 0
"""

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .colors import normalize_color_map, auto_color_map


PANEL_W = 280
POINT_SIZE = 2.5


class DatasetViewer:
    """Interactive viewer for torch_geometric Dataset objects.

    Keyboard shortcuts:
        Right arrow / L  →  next frame
        Left arrow  / H  →  previous frame
        R               →  reset camera
    """

    def __init__(self, dataset, label_cfg: dict | None = None, start_idx: int = 0):
        self.dataset = dataset
        self.current_idx = start_idx
        self._cached_pos: np.ndarray | None = None
        self._cached_labels: np.ndarray | None = None

        if label_cfg is not None:
            self.color_map = normalize_color_map(label_cfg["color_map"])
            self.semantic_map = {int(k): v for k, v in label_cfg.get("semantic_map", {}).items()}
        else:
            n = 32
            self.color_map = auto_color_map(n)
            self.semantic_map = {i: str(i) for i in range(n)}

        self._class_ids = sorted(self.semantic_map.keys())
        self.active_classes: set[int] = set(self._class_ids)

        # GUI widget refs — populated in _build_window
        self._window = None
        self._scene: gui.SceneWidget | None = None
        self._panel = None
        self._lbl_frame: gui.Label | None = None
        self._lbl_npts: gui.Label | None = None
        self._lbl_stats: gui.Label | None = None
        self._checkboxes: dict[int, gui.Checkbox] = {}
        self._mat = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def launch(dataset, label_cfg: dict | None = None, start_idx: int = 0) -> None:
        """Create the app and block until the window is closed."""
        app = gui.Application.instance
        app.initialize()
        viewer = DatasetViewer(dataset, label_cfg, start_idx)
        viewer._build_window()
        app.run()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        app = gui.Application.instance
        w = app.create_window("LiDAR Dataset Viewer", 1500, 900)
        self._window = w
        em = w.theme.font_size

        # ---- Material ----
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = POINT_SIZE
        self._mat = mat

        # ---- 3-D scene ----
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([0.08, 0.08, 0.08, 1.0])
        self._scene.set_on_key(self._on_key)

        # ---- Left panel ----
        panel = gui.Vert(int(0.4 * em), gui.Margins(int(0.6 * em)))

        # -- Navigation --
        panel.add_child(self._section_label("Navigation", em))

        self._lbl_frame = gui.Label("— / —")
        self._lbl_frame.text_color = gui.Color(0.85, 0.85, 0.85)
        panel.add_child(self._lbl_frame)

        self._lbl_npts = gui.Label("Points: —")
        self._lbl_npts.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(self._lbl_npts)

        nav_row = gui.Horiz(int(0.3 * em))
        btn_prev = gui.Button("◀  Prev")
        btn_prev.set_on_clicked(self._on_prev)
        btn_next = gui.Button("Next  ▶")
        btn_next.set_on_clicked(self._on_next)
        nav_row.add_stretch()
        nav_row.add_child(btn_prev)
        nav_row.add_child(btn_next)
        nav_row.add_stretch()
        panel.add_child(nav_row)

        panel.add_child(gui.Label(""))

        # -- Stats --
        panel.add_child(self._section_label("Class distribution", em))
        self._lbl_stats = gui.Label("—")
        self._lbl_stats.text_color = gui.Color(0.75, 0.75, 0.75)
        panel.add_child(self._lbl_stats)

        panel.add_child(gui.Label(""))

        # -- Class filter --
        panel.add_child(self._section_label("Filter classes", em))

        toggle_row = gui.Horiz(int(0.3 * em))
        btn_show_all = gui.Button("Show all")
        btn_show_all.set_on_clicked(self._on_show_all)
        btn_hide_all = gui.Button("Hide all")
        btn_hide_all.set_on_clicked(self._on_hide_all)
        toggle_row.add_child(btn_show_all)
        toggle_row.add_child(btn_hide_all)
        panel.add_child(toggle_row)

        scroll = gui.ScrollableVert(
            int(0.3 * em), gui.Margins(0, 0, int(0.3 * em), 0)
        )
        for cls_id in self._class_ids:
            name = self.semantic_map.get(cls_id, str(cls_id))
            cb = gui.Checkbox(f"{cls_id}: {name}")
            cb.checked = True
            cb.set_on_checked(
                lambda checked, cid=cls_id: self._on_class_toggle(cid, checked)
            )
            self._checkboxes[cls_id] = cb
            scroll.add_child(cb)

        panel.add_child(scroll)

        # ---- Layout ----
        w.add_child(self._scene)
        w.add_child(panel)
        self._panel = panel
        w.set_on_layout(self._on_layout)

        self._refresh()

    @staticmethod
    def _section_label(text: str, em: float) -> gui.Label:
        lbl = gui.Label(text.upper())
        lbl.text_color = gui.Color(0.5, 0.8, 1.0)
        return lbl

    def _on_layout(self, _ctx) -> None:
        r = self._window.content_rect
        self._panel.frame = gui.Rect(r.x, r.y, PANEL_W, r.height)
        self._scene.frame = gui.Rect(
            r.x + PANEL_W, r.y, r.width - PANEL_W, r.height
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_next(self) -> None:
        self.current_idx = (self.current_idx + 1) % len(self.dataset)
        self._refresh()

    def _on_prev(self) -> None:
        self.current_idx = (self.current_idx - 1) % len(self.dataset)
        self._refresh()

    def _on_key(self, event) -> int:
        if event.type == gui.KeyEvent.DOWN:
            if event.key in (gui.KeyName.RIGHT, ord("l"), ord("L")):
                self._on_next()
                return gui.Widget.EventCallbackResult.HANDLED
            if event.key in (gui.KeyName.LEFT, ord("h"), ord("H")):
                self._on_prev()
                return gui.Widget.EventCallbackResult.HANDLED
            if event.key in (ord("r"), ord("R")):
                self._reset_camera()
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_class_toggle(self, cls_id: int, checked: bool) -> None:
        if checked:
            self.active_classes.add(cls_id)
        else:
            self.active_classes.discard(cls_id)
        if self._cached_pos is not None:
            self._update_cloud(self._cached_pos, self._cached_labels)

    def _on_show_all(self) -> None:
        self.active_classes = set(self._class_ids)
        for cb in self._checkboxes.values():
            cb.checked = True
        if self._cached_pos is not None:
            self._update_cloud(self._cached_pos, self._cached_labels)

    def _on_hide_all(self) -> None:
        self.active_classes = set()
        for cb in self._checkboxes.values():
            cb.checked = False
        if self._cached_pos is not None:
            self._update_cloud(self._cached_pos, self._cached_labels)

    # ------------------------------------------------------------------
    # Data loading & rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        data = self.dataset.get(self.current_idx)
        pos = data.pos.numpy()
        labels = data.y.numpy()
        self._cached_pos = pos
        self._cached_labels = labels

        # Frame label
        suffix = ""
        if hasattr(data, "pcd_file") and data.pcd_file:
            suffix = f"\n{str(data.pcd_file).split('/')[-1]}"
        self._lbl_frame.text = f"{self.current_idx + 1} / {len(self.dataset)}{suffix}"
        self._lbl_npts.text = f"Points: {len(pos):,}"
        self._lbl_stats.text = self._build_stats_text(labels)

        self._update_cloud(pos, labels)

    def _update_cloud(self, pos: np.ndarray, labels: np.ndarray) -> None:
        mask = np.isin(labels, list(self.active_classes))
        pos_f = pos[mask].astype(np.float64)
        labels_f = labels[mask]

        default = [128, 128, 128]
        colors = np.array(
            [[c / 255.0 for c in self.color_map.get(int(l), default)] for l in labels_f],
            dtype=np.float64,
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos_f)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        scene = self._scene.scene
        scene.clear_geometry()
        if len(pos_f) > 0:
            scene.add_geometry("cloud", pcd, self._mat)
            self._reset_camera()

    def _reset_camera(self) -> None:
        if self._cached_pos is None:
            return
        mask = np.isin(self._cached_labels, list(self.active_classes))
        pos_f = self._cached_pos[mask].astype(np.float64)
        if len(pos_f) == 0:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos_f)
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _build_stats_text(self, labels: np.ndarray) -> str:
        total = max(len(labels), 1)
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(-counts)
        lines = []
        for cls_id, cnt in zip(unique[order], counts[order]):
            name = self.semantic_map.get(int(cls_id), str(cls_id))
            pct = 100.0 * cnt / total
            bar = "█" * int(pct / 5)
            lines.append(f"{name[:14]:<14} {pct:5.1f}% {bar}")
        return "\n".join(lines)
