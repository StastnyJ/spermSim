# grn_viz_qt.py
import sys
import math
import queue as queue_mod
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PyQt5.QtCore import QTimer, Qt, QPointF
from PyQt5.QtGui import QBrush, QPen, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
)

# ----------------------------
# Helpers
# ----------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def value_to_color(v: float) -> QColor:
    """
    Map [0..1] value to a heat-like color.
    Low -> dark/blue-ish, high -> bright/red-ish.
    """
    v = clamp01(v)
    r = int(255 * v)
    g = int(80 * (1.0 - abs(v - 0.5) * 2.0))  # a little mid bump
    b = int(255 * (1.0 - v))
    return QColor(r, g, b)

def weight_to_pen(w: float) -> QPen:
    """
    Positive = green-ish, negative = red-ish. Thickness by |w|.
    """
    mag = min(6.0, 0.5 + abs(w) * 6.0)
    if w >= 0:
        color = QColor(60, 180, 75)   # green
    else:
        color = QColor(220, 60, 60)   # red
    pen = QPen(color)
    pen.setWidthF(mag)
    pen.setCosmetic(True)
    return pen

# ----------------------------
# Visual items
# ----------------------------

@dataclass
class NodeVisual:
    circle: QGraphicsEllipseItem
    label: QGraphicsTextItem
    value_text: QGraphicsTextItem

@dataclass
class EdgeVisual:
    line: QGraphicsLineItem
    weight_text: QGraphicsTextItem

class GRNViewer(QGraphicsView):
    def __init__(self, state_queue, poll_hz: int = 30):
        super().__init__()

        self.state_queue = state_queue
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Track visuals
        self.nodes: Dict[str, NodeVisual] = {}
        self.edges: List[Tuple[str, str, float, EdgeVisual]] = []

        # self.setRenderHint(self.RenderHint.Antialiasing, True)
        self.setWindowTitle("GRN Visualizer (PyQt)")
        self.resize(900, 700)

        # Poll queue periodically
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_queue)
        self.timer.start(int(1000 / max(1, poll_hz)))

        # If we haven't built a graph yet, wait for first payload
        self.graph_built = False

    def build_graph(self, payload: dict):
        """
        Build node positions and edges once, from a payload:
        {
          "groups": {"inputs":[...], "internal":[...], "outputs":[...]},
          "edges": [(src, dst, w), ...],
          "values": {gene: value, ...}
        }
        """
        groups = payload["groups"]
        edges = payload["edges"]
        values = payload["values"]

        self.scene.clear()
        self.nodes.clear()
        self.edges.clear()

        # Layout: three columns (inputs, internal, outputs)
        cols = [("inputs", groups.get("inputs", [])),
                ("internal", groups.get("internal", [])),
                ("outputs", groups.get("outputs", []))]

        col_x = {"inputs": 100, "internal": 420, "outputs": 740}
        top_y = 80
        v_gap = 128
        radius = 26

        font_label = QFont("Arial", 10)
        font_value = QFont("Arial", 9)

        positions: Dict[str, QPointF] = {}

        for col_name, names in cols:
            y = col_x[col_name]
            for i, name in enumerate(names):
                x = top_y + i * v_gap
                positions[name] = QPointF(x, y)

        # Draw edges
        for (src, dst, w) in edges:
            if src not in positions or dst not in positions:
                continue

            p1 = positions[src]
            p2 = positions[dst]

            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line.setPen(weight_to_pen(float(w) * (values.get(src, 0.0) + 0.00001)))
            self.scene.addItem(line)

            # weight text near midpoint
            mx = (p1.x() + p2.x()) * 0.5
            my = (p1.y() + p2.y()) * 0.5
            wt = QGraphicsTextItem(f"{float(w):.2f}")
            wt.setFont(QFont("Arial", 8))
            wt.setDefaultTextColor(Qt.darkGray)
            wt.setPos(mx + 4, my + 2)
            self.scene.addItem(wt)

            self.edges.append((src, dst, float(w), EdgeVisual(line=line, weight_text=wt)))

        for col_name, names in cols:
            y = col_x[col_name]
            for i, name in enumerate(names):
                x = top_y + i * v_gap

                v = float(values.get(name, 0.0))
                color = value_to_color(v)

                circle = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
                circle.setBrush(QBrush(color))
                circle.setPen(QPen(Qt.black))
                circle.setPos(positions[name])
                self.scene.addItem(circle)

                label = QGraphicsTextItem(name)
                label.setFont(font_label)
                label.setDefaultTextColor(Qt.black)
                label.setPos(positions[name].x() - radius, positions[name].y() + radius + 4)
                self.scene.addItem(label)

                value_text = QGraphicsTextItem(f"{v:.2f}")
                value_text.setFont(font_value)
                value_text.setDefaultTextColor(Qt.black)
                value_text.setPos(positions[name].x() - 14, positions[name].y() - 10)
                self.scene.addItem(value_text)

                self.nodes[name] = NodeVisual(circle=circle, label=label, value_text=value_text)

        self.graph_built = True

    def update_values(self, values: Dict[str, float]):
        for name, vis in self.nodes.items():
            v = float(values.get(name, 0.0))
            vis.circle.setBrush(QBrush(value_to_color(v)))
            vis.value_text.setPlainText(f"{v:.2f}")

    def poll_queue(self):
        """
        Drain queue quickly; keep only the newest state for smooth UI.
        """
        latest = None
        try:
            while True:
                latest = self.state_queue.get_nowait()
        except queue_mod.Empty:
            pass

        if latest is None:
            return

        # Build once, then only update values
        if not self.graph_built:
            self.build_graph(latest)
        else:
            self.build_graph(latest)
            # self.update_values(latest.get("values", {}))

def run_visualizer(state_queue):
    app = QApplication(sys.argv)
    view = GRNViewer(state_queue)
    view.show()
    sys.exit(app.exec_())
