import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg

# load data
def load_curve_data(filename):
    data = {1: {'reward': [], 'loss': [], 'success_rate': []},
            4: {'reward': [], 'loss': [], 'success_rate': []},
            8: {'reward': [], 'loss': [], 'success_rate': []}}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            buf, typ, idx, val = parts
            try:
                buf = int(buf)
                idx = int(idx)
                val = float(val)
            except ValueError:
                continue
            if buf not in data or typ not in data[buf]:
                continue
            while len(data[buf][typ]) <= idx:
                data[buf][typ].append(None)
            data[buf][typ][idx] = val
    for buf in data:
        for typ in data[buf]:
            data[buf][typ] = [v for v in data[buf][typ] if v is not None]
    return data

colors = {1: 'r', 4: 'g', 8: 'b'}

class PlotApp(QtWidgets.QWidget):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.setWindowTitle("ReplayBuffer Loss Curve (PyQtGraph)")

        layout = QtWidgets.QVBoxLayout(self)

        # 单图：只显示 loss
        self.plot = pg.PlotWidget()
        self.plot.setLabel('left', 'Loss')
        self.plot.setLabel('bottom', 'Index')
        self.plot.addLegend()
        layout.addWidget(self.plot)

        # 按钮：切换高亮的 buffer
        btn_layout = QtWidgets.QHBoxLayout()
        for rb in [1, 4, 8]:
            btn = QtWidgets.QPushButton(f"ReplayBuffer {rb}")
            btn.clicked.connect(lambda _, rb=rb: self.plot_data(rb))
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        self.plot_data(1)

    def plot_data(self, rb):
        self.plot.clear()
        self.plot.addLegend()

        for buf in [1, 4, 8]:
            y = self.data[buf]['loss']
            x = list(range(len(y)))
            width = 3 if buf == rb else 1
            alpha = 255 if buf == rb else 100
            color = pg.mkColor(colors[buf])
            color.setAlpha(alpha)
            pen = pg.mkPen(color=color, width=width)
            self.plot.plot(x, y, pen=pen, name=f'buffer={buf}')

if __name__ == '__main__':
    print("Loading curve data...")
    data = load_curve_data('curve_data.txt')
    print("Curve data loaded successfully.")

    app = QtWidgets.QApplication(sys.argv)
    win = PlotApp(data)
    win.resize(900, 400)
    win.show()
    sys.exit(app.exec_())