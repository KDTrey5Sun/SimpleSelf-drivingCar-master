import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# 手动修改为你的数据文件路径
# DATA_PATH = '/Users/smz/Desktop/SimpleSelf-drivingCar-master/curve_data.txt'
# DATA_PATH = './v3/curve_data_copy.txt'
DATA_PATH = './v3_pro_windowless/batch_size_data/curve_data_batchsize.txt'

def load_curve_data(filename):
    """
    读取: buffer,metric,index,value
    自动收集所有 buffer 和 metric，并按 index 聚合为序列。
    返回: data(dict), metrics(list), buffers(list)
      data[buf][metric] -> list[float]
    """
    data = {}
    metrics_set = set()
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            buf_str, typ, idx_str, val_str = parts
            try:
                buf = int(buf_str)
                idx = int(idx_str)
                val = float(val_str)
            except ValueError:
                continue

            if buf not in data:
                data[buf] = {}
            if typ not in data[buf]:
                data[buf][typ] = []
            metrics_set.add(typ)

            while len(data[buf][typ]) <= idx:
                data[buf][typ].append(None)
            data[buf][typ][idx] = val

    # 压紧去 None
    for buf in data:
        for typ in data[buf]:
            data[buf][typ] = [v for v in data[buf][typ] if v is not None]

    preferred = ['reward', 'loss', 'success_rate']
    metrics = [m for m in preferred if m in metrics_set] + \
              [m for m in sorted(metrics_set) if m not in preferred]
    buffers = sorted(data.keys())
    return data, metrics, buffers

def make_color_map(buffers):
    n = max(len(buffers), 1)
    colors = {}
    for i, buf in enumerate(buffers):
        colors[buf] = pg.intColor(i, hues=n, alpha=255)
    return colors

class PlotApp(QtWidgets.QWidget):
    def __init__(self, data, metrics, buffers):
        super().__init__()
        self.data = data
        self.metrics = metrics
        self.buffers = buffers
        self.colors = make_color_map(self.buffers)

        self.setWindowTitle("ReplayBuffer Curve Visualization (Auto-detected)")
        root = QtWidgets.QVBoxLayout(self)

        # 多个指标图
        self.plots = []
        for metric in self.metrics:
            pw = pg.PlotWidget()
            pw.setLabel('left', metric)
            pw.setLabel('bottom', 'Index')
            pw.addLegend()
            root.addWidget(pw)
            self.plots.append(pw)

        # 按钮区域：放入可横向滚动容器
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        # Some PyQt type stubs may not expose ScrollBarAsNeeded/AlwaysOff on Qt
        h_policy = getattr(QtCore.Qt, 'ScrollBarAsNeeded', QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        v_policy = getattr(QtCore.Qt, 'ScrollBarAlwaysOff', QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(h_policy)
        scroll.setVerticalScrollBarPolicy(v_policy)
        btn_host = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_host)
        btn_layout.setContentsMargins(8, 8, 8, 8)
        btn_layout.setSpacing(8)

        # “Show All”按钮
        show_all_btn = QtWidgets.QPushButton("Show All")
        show_all_btn.clicked.connect(self.plot_all)
        btn_layout.addWidget(show_all_btn)

        # 动态生成每个 buffer 的按钮
        for rb in self.buffers:
            btn = QtWidgets.QPushButton(f"ReplayBuffer {rb}")
            btn.clicked.connect(lambda _, rb=rb: self.plot_data(rb))
            btn_layout.addWidget(btn)

        btn_layout.addStretch(1)
        scroll.setWidget(btn_host)
        root.addWidget(scroll)

        # 默认显示全部并高亮第一个
        self.plot_all(highlight=self.buffers[0] if self.buffers else None)

    def plot_all(self, highlight=None):
        for i, metric in enumerate(self.metrics):
            self.plots[i].clear()
            self.plots[i].addLegend()
            for buf in self.buffers:
                y = self.data.get(buf, {}).get(metric, [])
                x = list(range(len(y)))
                width = 3 if (highlight is not None and buf == highlight) else 1
                alpha = 255 if (highlight is not None and buf == highlight) else 140
                c = pg.mkColor(self.colors[buf])
                c.setAlpha(alpha)
                pen = pg.mkPen(color=c, width=width)
                self.plots[i].plot(x, y, pen=pen, name=f'buffer={buf}')

    def plot_data(self, rb):
        for i, metric in enumerate(self.metrics):
            self.plots[i].clear()
            self.plots[i].addLegend()
            for buf in self.buffers:
                y = self.data.get(buf, {}).get(metric, [])
                x = list(range(len(y)))
                width = 3 if buf == rb else 1
                alpha = 255 if buf == rb else 100
                c = pg.mkColor(self.colors[buf])
                c.setAlpha(alpha)
                pen = pg.mkPen(color=c, width=width)
                self.plots[i].plot(x, y, pen=pen, name=f'buffer={buf}')

if __name__ == '__main__':
    print(f"Loading curve data from: {DATA_PATH}")
    data, metrics, buffers = load_curve_data(DATA_PATH)
    print(f"Detected buffers: {buffers}")
    print(f"Detected metrics: {metrics}")

    app = QtWidgets.QApplication(sys.argv)
    win = PlotApp(data, metrics, buffers)
    win.resize(1100, 320 + 260 * max(1, len(metrics)))  # 适配高度
    win.show()
    sys.exit(app.exec_())