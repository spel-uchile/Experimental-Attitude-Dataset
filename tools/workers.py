"""
Created by Elias Obreque
els.obrq@gmail.com
Date: 18-10-2022
"""
from qtpy import QtWidgets
from qtpy.QtCore import Signal, QObject, QTimer
from threading import Thread


class ViewWorker(QObject, Thread):
    started = Signal()
    finished = Signal()
    failed = Signal(str)

    def __init__(self, target, name=None, args=(), finished_callback=None, daemon=True):
        QObject.__init__(self, None)
        Thread.__init__(self, target=target, name=name, args=args, daemon=daemon)
        if finished_callback:
            self.finished.connect(finished_callback)

    def run(self):
        try:
            super(ViewWorker, self).run()
            self.finished.emit()
        except Exception as e:
            self.failed.emit("{} failed: {}".format(self.name, str(e)))


class MathWorker(QObject, Thread):
    started = Signal()
    finished = Signal()
    failed = Signal(str)

    def __init__(self, target, name=None, args=(), finished_callback=None):
        QObject.__init__(self, None)
        Thread.__init__(self, target=target, name=name, args=args)
        if finished_callback:
            self.finished.connect(finished_callback)

    def run(self):
        try:
            self.started.emit()
            super(MathWorker, self).run()
            self.finished.emit()
        except Exception as e:
            print("{} failed: {}".format(self.name, str(e)))
            self.failed.emit("{} failed: {}".format(self.name, str(e)))


class StatusBar(QObject):
    progress_signal = Signal(int)

    def __init__(self, progress_bar: QtWidgets.QProgressBar, label_msg: QtWidgets.QLabel):
        super(StatusBar, self).__init__()
        self.progress_bar = progress_bar
        self.label = label_msg
        self.progress_signal.connect(self._show_temporally)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(5*1000)
        self._timer.timeout.connect(self.reset)

    def reset(self):
        self.label.setText("")
        self.label.setStyleSheet("QLineEdit{background: white;}")

    def _show_temporally(self, progress, timer=True):
        if progress is not None:
            self.progress_bar.setValue(progress)
        if timer:
            self._timer.start()

    def send_value(self, progress, timer=True):
        self.progress_signal.emit(progress)


