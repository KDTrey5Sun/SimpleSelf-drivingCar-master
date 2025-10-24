import sys
import os
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt

# 手动修改为你的数据文件路径
# DATA_PATH = '/Users/smz/Desktop/SimpleSelf-drivingCar-master/curve_data.txt'
# DATA_PATH = './v3/curve_data_copy.txt'
# 默认使用 buffersize 实验生成的文件
DATA_PATH = './v3_pro_windowless/memory_size_data/curve_data_buffersize.txt'

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

    preferred = ['reward', 'loss', 'epsilon', 'success_rate']
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
            pw.setLabel('bottom', 'Episode')
            pw.addLegend()
            root.addWidget(pw)
            self.plots.append(pw)

        # 按钮区域：放入可横向滚动容器
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
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


def save_matplotlib_grid(data, metrics, buffers, out_path):
    """
    使用 Matplotlib 画一个网格图：每个 metric 一张子图，每条曲线是一个 buffer。
    """
    import math
    n_metrics = len(metrics)
    cols = 2
    rows = math.ceil(n_metrics / cols)
    plt.figure(figsize=(12, 3.2 + 2.6 * max(1, rows)))
    cmap = plt.get_cmap('tab10')

    for i, metric in enumerate(metrics):
        ax = plt.subplot(rows, cols, i + 1)
        for j, buf in enumerate(buffers):
            y = data.get(buf, {}).get(metric, [])
            x = list(range(len(y)))
            ax.plot(x, y, label=f'buffer={buf}', color=cmap(j % 10), linewidth=1.8)
        ax.set_title(metric)
        ax.set_xlabel('Episode')
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)

    # 如果子图数量是奇数，隐藏最后一个空子图
    total = rows * cols
    if n_metrics < total:
        for k in range(n_metrics + 1, total + 1):
            ax = plt.subplot(rows, cols, k)
            ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"Saved figure to: {out_path}")

if __name__ == '__main__':
    # 允许通过命令行传入数据路径，第二个可选参数为 --save 或输出文件路径
    argv = sys.argv[1:]
    data_path = DATA_PATH
    out_png = None
    if len(argv) >= 1 and argv[0] not in ('--save', '-s'):
        data_path = argv[0]
    if len(argv) >= 1 and argv[0] in ('--save', '-s'):
        out_png = './v3_pro_windowless/memory_size_data/curve_plots.png'
    if len(argv) >= 2:
        if argv[1] in ('--save', '-s'):
            out_png = './v3_pro_windowless/memory_size_data/curve_plots.png'
        else:
            out_png = argv[1]

    if not os.path.exists(data_path):
        print(f"[Error] Data file not found: {data_path}")
        print("Hint: pass the correct file path as the first argument, e.g.\n  python curve_data_plot.py ./v3_pro_windowless/memory_size_data/curve_data_buffersize.txt [--save or out.png]")
        sys.exit(1)

    print(f"Loading curve data from: {data_path}")
    data, metrics, buffers = load_curve_data(data_path)
    print(f"Detected buffers: {buffers}")
    print(f"Detected metrics: {metrics}")

    # 仅保存图片（无 GUI）
    if out_png is not None:
        save_matplotlib_grid(data, metrics, buffers, out_png)
        sys.exit(0)

    app = QtWidgets.QApplication(sys.argv)
    win = PlotApp(data, metrics, buffers)
    win.resize(1100, 320 + 260 * max(1, len(metrics)))  # 适配高度
    win.show()
    sys.exit(app.exec_())