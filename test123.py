import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg

app = QtWidgets.QApplication(sys.argv)
win = pg.plot([1, 3, 2, 4, 3, 5])
win.setWindowTitle("PyQt5 + PyQtGraph 测试")
sys.exit(app.exec_())
