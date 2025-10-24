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
            buf = int(buf)
            if buf not in data or typ not in data[buf]:
                continue
            idx = int(idx)
            val = float(val)
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
        self.setWindowTitle("ReplayBuffer Curve Visualization (PyQtGraph)")

        layout = QtWidgets.QVBoxLayout(self)

        self.plots = []
        metrics = ['reward', 'loss', 'success_rate']
        ylabels = ['Reward', 'Loss', 'Success Rate']

        for i in range(3):
            pw = pg.PlotWidget()
            pw.setLabel('left', ylabels[i])
            pw.setLabel('bottom', 'Index')
            pw.addLegend()
            layout.addWidget(pw)
            self.plots.append(pw)

        btn_layout = QtWidgets.QHBoxLayout()
        for rb in [1, 4, 8]:
            btn = QtWidgets.QPushButton(f"ReplayBuffer {rb}")
            btn.clicked.connect(lambda _, rb=rb: self.plot_data(rb))
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        self.plot_data(1)

    def plot_data(self, rb):
        metrics = ['reward', 'loss', 'success_rate']

        for i, metric in enumerate(metrics):
            self.plots[i].clear()
            self.plots[i].addLegend()

            for buf in [1, 4, 8]:
                y = self.data[buf][metric]
                x = list(range(len(y)))
                width = 3 if buf == rb else 1
                alpha = 255 if buf == rb else 100  
                color = pg.mkColor(colors[buf])
                color.setAlpha(alpha)
                pen = pg.mkPen(color=color, width=width)
                self.plots[i].plot(x, y, pen=pen, name=f'buffer={buf}')



if __name__ == '__main__':
    print("Loading curve data...")
    data = load_curve_data('curve_data.txt')
    print("Curve data loaded successfully.")

    app = QtWidgets.QApplication(sys.argv)
    win = PlotApp(data)
    win.resize(900, 800)
    win.show()
    sys.exit(app.exec_())
