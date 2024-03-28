#!/usr/bin/python3.7.4
# -*- coding:utf-8 -*-
# @Author: 叶不落
# @Time: 2024/3/1510:17
# @desc:实现业务逻辑，调用UI类

import math
import numpy as np
import logging
from multiprocessing import Process, Queue
from typing import List
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox, QGraphicsScene,QGraphicsItem,
                             QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsEllipseItem)
from PyQt5.QtGui import QIcon, QTextCursor, QPixmap, QImage, QCursor, QPen, QColor, QBrush, QPainter,QPainterPath
from PyQt5.QtCore import QEvent, QRectF, Qt, QObject, pyqtSignal, QThread,QPointF
from new_MainWindow import Ui_MainWindow
import dic_algorithm
import resource_rc

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体，中文
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像时负号（中文）显示为方块的问题


# 日志处理器
class QPlainTextEditLogger(logging.Handler, QObject):
    append_signal = pyqtSignal(str)

    def __init__(self, widget):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.widget = widget
        self.widget.setReadOnly(True)
        self.append_signal.connect(self.append_text)

    def emit(self, record):
        if record:
            msg = self.format(record)
            self.append_signal.emit(msg)  # 发送自定义信号

    def append_text(self, message):
        if message:
            self.widget.appendPlainText(message)
            self.widget.moveCursor(QTextCursor.End)


# 日志监听线程
class LogListenerThread(QThread):
    new_log_signal = pyqtSignal(logging.LogRecord)  # 用于从子进程接收LogRecord对象的信号

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.running = True

    def run(self):
        while self.running:
            record = self.log_queue.get(10)
            self.new_log_signal.emit(record)

    def stop(self):
        self.running = False


class MyWin(QMainWindow):
    def __init__(self, parent=None):
        super(MyWin, self).__init__(parent=None)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.tabWidget.setCurrentIndex(0)  # 确保前处理页面在最开始
        self.ui.graphicsView_cor_img.viewport().installEventFilter(self)
        self.ui.graphicsView_post_img.viewport().installEventFilter(self)
        # 一些初始属性，如路径等
        self.img_path = ''
        self.camera_path = ''
        self.result_save_path = ''
        self.input_img_method = ''
        self.pre_process_method = ''
        self.img_list = None  # cv读取出来的二维矩阵组成列表
        self.subset_size = 31
        self.step = 5
        self.roi_top_left_point: np.ndarray = None  # 感兴趣区域左上角点,一维数组
        self.roi_bottom_right_point: np.ndarray = None
        self.cor_criterion = ''
        self.shap_func = ''
        self.int_search_method = ''
        self.inter_method = ''
        self.sub_search_method = ''
        self.is_reliable = True
        self.result_input_path = ''
        self.img_input_path = ''
        self.is_saved = False
        self.strain_type = ''
        self.strain_com_method = ''

        self.dic_list_img: List[dic_algorithm.DIC] = []
        self.x_points = None  # 计算点x坐标，一维数组
        self.y_points = None

        # 一些标志信号，反馈算法处理情况
        self.pre_is_ok = False
        self.cor_is_ok = False
        self.post_is_ok = False
        self.draw_roi = False  # 是否允许进行ROI框选
        self.roi_is_ok = False
        self.dic_is_ok = False
        self.strain_is_ok = False
        self.dic_queue = Queue()
        self.strain_queue = Queue()
        self.log_queue = Queue()

        # 加载信号的初始化
        self.setup()
        self.showMaximized()

        self.set_logger()  # 初始化日志记录器

    def set_logger(self):
        # 设置日志记录器
        logging.basicConfig(level=logging.INFO)
        self.logger1 = logging.getLogger('前处理')
        self.logger2 = logging.getLogger('相关处理')
        self.logger3 = logging.getLogger('后处理')
        self.formatter = logging.Formatter('%(asctime)s ------- %(message)s', datefmt='%m-%d %H:%M')  # 日志格式
        fileHandler = logging.FileHandler('日志记录.log')  # 汇总文件日志记录
        fileHandler.setFormatter(self.formatter)
        # 针对前处理页面日志
        logHandler1 = QPlainTextEditLogger(self.ui.plainTextEdit_pre_message)
        logHandler1.setFormatter(self.formatter)
        self.logger1.addHandler(fileHandler)
        self.logger1.addHandler(logHandler1)
        # 针对相关处理页面日志
        self.logHandler2 = QPlainTextEditLogger(self.ui.plainTextEdit_cor_message)
        self.logHandler2.setFormatter(self.formatter)
        self.logger2.addHandler(fileHandler)
        self.logger2.addHandler(self.logHandler2)
        # 针对后处理页面日志
        self.logHandler3 = QPlainTextEditLogger(self.ui.plainTextEdit_post_message)
        self.logHandler3.setFormatter(self.formatter)
        self.logger3.addHandler(fileHandler)
        self.logger3.addHandler(self.logHandler3)

        # 初始化日志监听线程
        self.cor_listener_thread = LogListenerThread(self.log_queue)
        self.cor_listener_thread.new_log_signal.connect(self.logHandler2.emit)
        self.post_listener_thread = LogListenerThread(self.log_queue)
        self.post_listener_thread.new_log_signal.connect(self.logHandler3.emit)

    def setup(self):
        self.ui.pushButton_pre_ok.clicked.connect(self.do_pushButton_pre_ok_clicked)
        self.ui.pushButton_result_save_path.clicked.connect(self.do_pushButton_result_save_path_clicked)
        self.ui.pushButton_pre_clear.clicked.connect(self.do_pushButton_pre_clear_clicked)

        self.ui.tabWidget.currentChanged.connect(self.do_tabWidget_currentChanged)
        self.ui.pushButton_roi_start.clicked.connect(self.do_pushButton_roi_start_clicked)
        self.ui.pushButton_roi_clear.clicked.connect(self.do_pushButton_roi_clear_clicked)
        self.ui.pushButton_roi_select_all.clicked.connect(self.do_pushButton_roi_all_clicked)
        self.ui.pushButton_roi_ok.clicked.connect(self.do_pushButton_roi_ok_clicked)
        self.ui.pushButton_cor_params_ok.clicked.connect(self.do_pushButton_cor_params_ok_clicked)
        self.ui.pushButton_cor_params_clear.clicked.connect(self.do_pushButton_cor_params_clear_clicked)

        self.ui.pushButton_result_input_path.clicked.connect(self.do_pushButton_result_input_path_clicked)
        self.ui.pushButton_input_img_path.clicked.connect(self.do_pushButton_input_img_path_clicked)
        self.ui.pushButton_post_clear.clicked.connect(self.do_pushButton_post_clear_clicked)
        self.ui.pushButton_post_ok.clicked.connect(self.do_pushButton_post_ok_clicked)
        self.ui.pushButton_save_img.clicked.connect(self.do_pushButton_save_img_clicked)
        self.ui.comboBox_img_type.currentIndexChanged.connect(self.do_comboBox_img_currentIndexChanged)
        self.ui.comboBox_img_color_type.currentIndexChanged.connect(self.do_comboBox_img_currentIndexChanged)

    # 前处理tab1页面进行信号处理
    def do_pushButton_result_save_path_clicked(self):
        fileDialog = QFileDialog(self)
        fileDialog.setWindowIcon(QIcon(":/image/ui_image/选取文件.png"))
        self.result_save_path = fileDialog.getExistingDirectory(self, '选择结果文件保存的目录', '')

    def do_pushButton_pre_clear_clicked(self):
        self.ui.comboBox_input_img_method.setCurrentIndex(0)
        self.ui.comboBox_pre_process_method.setCurrentIndex(0)
        self.img_path = ''
        self.camera_path = ''
        self.result_save_path = ''

    def do_pushButton_pre_ok_clicked(self):
        fileDialog = QFileDialog(self)
        if self.ui.comboBox_input_img_method.currentIndex() == 1:
            camera_path, _ = fileDialog.getOpenFileName(self, "请选择数字相机的可执行程序。", "",
                                                        "Executable Files (*.exe)")
            if camera_path:
                img_path, _ = fileDialog.getExistingDirectory(self, '请选择数字相机存储图像的目录。')
                if img_path:
                    self.camera_path = camera_path
                    self.img_path = img_path
        else:
            img_path = fileDialog.getExistingDirectory(self, '请选择图像或视频所在目录,注意路径中不能有空格！')
            self.img_path = img_path

        try:
            if self.ui.comboBox_input_img_method.currentIndex() == 2:
                img_list: List[np.ndarray] = dic_algorithm.pre_process(False, self.img_path,
                                                                       self.ui.comboBox_pre_process_method.currentText())
            else:
                self.img_list: List[np.ndarray] = dic_algorithm.pre_process(True, self.img_path,
                                                                            self.ui.comboBox_pre_process_method.currentText())
            ref_img = self.img_list[0]
            # 转换为QImage对象，注意灰度图像的bytes_per_line参数
            height, width = ref_img.shape
            bytes_per_line = width  # 对于灰度图像，每个像素只有一个字节
            q_image = QImage(ref_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            # 创建QPixmap对象
            pixmap = QPixmap.fromImage(q_image)
            # 创建场景并添加图像项
            scene = QGraphicsScene(self)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.addItem(pixmap_item)
            # 将场景设置到QGraphicsView
            self.ui.graphicsView_pre_img.setScene(scene)  # self.graphicsView应该是UI文件中的QGraphicsView
            self.pre_is_ok = True  # 前处理成功
        except Exception as e:
            self.logger1.exception(e)

        # 打印日志
        self.logger1.info('***欢迎使用前处理器，以下为您设置好的参数：***')
        self.logger1.info(f'图像输入方法选择为：{self.ui.comboBox_input_img_method.currentText()}')
        self.logger1.info(f'前处理算法选择为：{self.ui.comboBox_pre_process_method.currentText()}')
        self.logger1.info(f'图像来源路径为：{self.img_path}')
        self.logger1.info(f'数字相机可执行程序为：{self.camera_path}')
        self.logger1.info(f'计算结果保存目录为：{self.result_save_path}')
        self.logger1.info('请确认是否无误，有误请按clear按钮，无误请来到匹配参数设置页面继续进行。')

    # 对匹配参数处理页面tab2进行信号处理
    def do_tabWidget_currentChanged(self, index):
        if not self.pre_is_ok and index == 1:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("建议先在前处理页面完成相关配置，处理成功后再进入本页面")
            msgBox.setWindowTitle("警告")
            msgBox.setWindowIcon(QIcon(":/image/ui_image/警告.png"))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msgBox.exec_()
            return
        if (not self.pre_is_ok or not self.cor_is_ok) and index == 2:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("若无已计算好的结果请回到前页面进行相关设置处理")
            msgBox.setWindowTitle("警告")
            msgBox.setWindowIcon(QIcon(":/image/ui_image/警告.png"))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msgBox.exec_()
            return
        if self.pre_is_ok and not self.cor_is_ok and index == 1:
            self.cor_listener_thread.start()

            ref_img = self.img_list[0]
            height, width = ref_img.shape
            bytes_per_line = width  # 对于灰度图像，每个像素只有一个字节
            q_image = QImage(ref_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            self.pixmap = QPixmap.fromImage(q_image)
            self.scene = QGraphicsScene(self)
            self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
            self.pixmap_item.setFlag(QGraphicsPixmapItem.ItemIsMovable)
            self.scene.addItem(self.pixmap_item)
            self.ui.graphicsView_cor_img.setScene(self.scene)
            return
        if self.pre_is_ok and self.cor_is_ok and index == 2:
            self.ui.graphicsView_post_img.setScene(self.scene)
            # 创建一个钢笔和画刷来绘制点，设置颜色和风格
            pen = QPen(QColor("red"))  # 描边颜色
            pen.setWidth(2)  # 描边粗细
            brush = QBrush(QColor("blue"))  # 填充颜色
            for x in self.x_points:
                for y in self.y_points:
                    ellipse_item = QGraphicsEllipseItem(x - 1, y - 1, 2, 2, self.pixmap_item)  # 设置父对象为图像
                    ellipse_item.setPen(pen)
                    ellipse_item.setBrush(brush)

            return

    def do_pushButton_roi_start_clicked(self):
        self.draw_roi = True
        self.logger2.info('请开始在右侧图像上框选感兴趣区域。')

    def eventFilter(self, obj, event):
        # 事件过滤器
        if obj is self.ui.graphicsView_cor_img.viewport():
            if self.draw_roi:
                if event.type() == QEvent.MouseButtonPress:
                    self.setCursor(QCursor(Qt.CrossCursor))
                    self.roi_top_left_point = np.array((self.ui.graphicsView_cor_img.mapToScene(event.pos()).x(),
                                                        self.ui.graphicsView_cor_img.mapToScene(event.pos()).y()))
                    return True
                elif event.type() == QEvent.MouseMove:
                    return True
                elif event.type() == QEvent.MouseButtonRelease:
                    self.roi_bottom_right_point = np.array((self.ui.graphicsView_cor_img.mapToScene(event.pos()).x(),
                                                            self.ui.graphicsView_cor_img.mapToScene(event.pos()).y()))
                    # 减去图像左上角点在场景中的坐标
                    self.roi_top_left_point -= np.array((self.scene.items()[0].pos().x(),
                                                         self.scene.items()[0].pos().y())).astype(int)
                    self.roi_bottom_right_point -= np.array((self.scene.items()[0].pos().x(),
                                                             self.scene.items()[0].pos().y())).astype(int)
                    self.logger2.info(
                        f'您所选择的ROI区域：左上角-{self.roi_top_left_point},\n右下角-{self.roi_bottom_right_point}')
                    cor_rect = QGraphicsRectItem()
                    cor_rect.setParentItem(self.pixmap_item)
                    cor_rect.setPen(QPen(Qt.red, 1))  # 红色粗1px边框
                    cor_rect.setRect(QRectF(*self.roi_top_left_point,
                                            self.roi_bottom_right_point[0] - self.roi_top_left_point[0],
                                            self.roi_bottom_right_point[1] - self.roi_top_left_point[1]))
                    self.scene.addItem(cor_rect)
                    self.setCursor(QCursor(Qt.ArrowCursor))
                    return True

            if self.roi_is_ok:
                if event.type() == QEvent.Wheel:
                    scaleFactor = 1.15  # 缩放因子，您可以根据需要调整此值
                    # 向前滚动放大
                    if event.angleDelta().y() > 0:
                        self.ui.graphicsView_cor_img.scale(scaleFactor, scaleFactor)
                    # 向后滚动缩小
                    elif event.angleDelta().y() < 0:
                        self.ui.graphicsView_cor_img.scale(1 / scaleFactor, 1 / scaleFactor)
                    return True

            return False

        if obj is self.ui.graphicsView_post_img.viewport():
            if self.strain_is_ok:
                if event.type() == QEvent.Wheel:
                    scaleFactor = 1.15  # 缩放因子，您可以根据需要调整此值
                    # 向前滚动放大
                    if event.angleDelta().y() > 0:
                        self.ui.graphicsView_post_img.scale(scaleFactor, scaleFactor)
                    # 向后滚动缩小
                    elif event.angleDelta().y() < 0:
                        self.ui.graphicsView_post_img.scale(1 / scaleFactor, 1 / scaleFactor)
                    return True
            return False

    def do_pushButton_roi_clear_clicked(self):
        self.roi_top_left_point = None
        self.roi_bottom_right_point = None
        for item in self.pixmap_item.childItems():
            self.scene.removeItem(item)
        self.logger2.info('已清除框选区域，请重新框选。')

    def do_pushButton_roi_all_clicked(self):
        self.roi_top_left_point = np.array((0, 0))
        self.roi_bottom_right_point = np.array((self.pixmap.width() - 1, self.pixmap.height() - 1)).astype(int)
        self.logger2.info(
            f'已框选全部区域，您所选择的ROI区域：\n左上角-{self.roi_top_left_point},右上角-{self.roi_bottom_right_point}')
        for item in self.pixmap_item.childItems():
            self.scene.removeItem(item)

    def do_pushButton_roi_ok_clicked(self):
        if isinstance(self.roi_bottom_right_point, np.ndarray):
            self.roi_top_left_point.astype(int)
            self.roi_bottom_right_point.astype(int)
            # 判断框选的点是否符合要求，在图像边界内
            if (self.roi_top_left_point[0] >= 0 and self.roi_top_left_point[1] >= 0
                    and self.roi_bottom_right_point[0] <= self.pixmap.width()
                    and self.roi_bottom_right_point[1] <= self.pixmap.height()):
                self.draw_roi = False
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Information)
                msgBox.setText("框选成功，请设置后续匹配参数！")
                msgBox.setWindowTitle("恭喜")
                msgBox.setWindowIcon(QIcon(":/image/ui_image/祝贺.png"))
                msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
                msgBox.exec_()
                self.subset_size = int(self.ui.spinBox_roi_subset_size.value())
                self.step = int(self.ui.spinBox_roi_step.value())
                self.roi_is_ok = True

                # 根据计算步长和矩形区域绘制网格
                line_width = 1
                line_color = Qt.red
                # 计算网格点
                self.x_points = np.arange(self.roi_top_left_point[0] + self.subset_size // 2,
                                          self.roi_bottom_right_point[0] - self.subset_size // 2, self.step).astype(int)
                self.y_points = np.arange(self.roi_top_left_point[1] + self.subset_size // 2,
                                          self.roi_bottom_right_point[1] - self.subset_size // 2, self.step).astype(int)
                # 绘制水平网格线
                pen = QPen(line_color, line_width)
                for y in self.y_points:
                    line = QGraphicsLineItem(self.roi_top_left_point[0], y, self.roi_bottom_right_point[0], y)
                    line.setParentItem(self.pixmap_item)
                    line.setPen(pen)
                    self.scene.addItem(line)
                # 绘制垂直网格线
                for x in self.x_points:
                    line = QGraphicsLineItem(x, self.roi_top_left_point[1], x, self.roi_bottom_right_point[1])
                    line.setParentItem(self.pixmap_item)
                    line.setPen(pen)
                    self.scene.addItem(line)
            else:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText("框选失败，返回点坐标在边界外，请重新框选！")
                msgBox.setWindowTitle("失败")
                msgBox.setWindowIcon(QIcon(":/image/ui_image/信息.png"))
                msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
                msgBox.exec_()
                self.logger2.error('ROI区域设置失败，必须重新返回框选！')
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("框选失败，未返回点坐标，请重新框选！")
            msgBox.setWindowTitle("失败")
            msgBox.setWindowIcon(QIcon(":/image/ui_image/信息.png"))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()
            self.logger2.error('ROI区域设置失败，必须重新返回框选！')

    def do_pushButton_cor_params_clear_clicked(self):
        self.ui.comboBox_correlate.setCurrentIndex(0)
        self.ui.comboBox_shape_func.setCurrentIndex(0)
        self.ui.comboBox_int_search_method.setCurrentIndex(0)
        self.ui.comboBox_inter_method.setCurrentIndex(0)
        self.ui.comboBox_sub_search_method.setCurrentIndex(0)
        self.ui.radioButton_is_reliable.setChecked(True)
        self.cor_criterion = ''
        self.shap_func = ''
        self.int_search_method = ''
        self.inter_method = ''
        self.sub_search_method = ''
        self.is_reliable = ''
        self.logger2.info('已经清除除ROI外所有匹配参数，请重新选择！')

    @staticmethod
    def dic_compute_displacement(queue, img_list, roi_top_left_point, roi_bottom_right_point, subset_size, step,
                                 inter_method,
                                 cor_criterion, int_search_method, shap_func, sub_search_method, is_reliable,
                                 log_queue):
        dic_list = dic_algorithm.correlate_parameter_compute(img_list, roi_top_left_point, roi_bottom_right_point,
                                                             subset_size, step, inter_method, cor_criterion,
                                                             int_search_method, shap_func, sub_search_method,
                                                             is_reliable,
                                                             log_queue=log_queue)
        queue.put(dic_list)

    def do_pushButton_cor_params_ok_clicked(self):
        if self.roi_is_ok:
            self.cor_criterion = self.ui.comboBox_correlate.currentText()
            self.shap_func = self.ui.comboBox_shape_func.currentText()
            self.int_search_method = self.ui.comboBox_int_search_method.currentText()
            self.inter_method = self.ui.comboBox_inter_method.currentText()
            self.sub_search_method = self.ui.comboBox_sub_search_method.currentText()
            self.is_reliable = self.ui.radioButton_is_reliable.isChecked()
            self.logger2.info('***匹配参数设置如下：***')
            self.logger2.info(f'感兴趣区域：{self.roi_top_left_point}-->{self.roi_bottom_right_point}')
            self.logger2.info(f'选取相关函数为：{self.cor_criterion}')
            self.logger2.info(f'选取形函数为：{self.shap_func}')
            self.logger2.info(f'选取整数搜索算法为：{self.int_search_method}')
            self.logger2.info(f'选取灰度插值算法为：{self.inter_method}')
            self.logger2.info(f'选取亚像素搜索算法为：{self.sub_search_method}')
            self.logger2.info(f'是否根据可靠性引导：{self.is_reliable}')

            # 开始进行DIC匹配算法
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("所有匹配参数已经完成，开始进行DIC算法匹配！")
            msgBox.setWindowTitle("恭喜")
            msgBox.setWindowIcon(QIcon(":/image/ui_image/祝贺.png"))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()
            self.cor_is_ok = True

            self.process = Process(target=self.dic_compute_displacement,args=(
                                   self.dic_queue, self.img_list, self.roi_top_left_point, self.roi_bottom_right_point,
                                   self.subset_size, self.step, self.inter_method, self.cor_criterion,
                                   self.int_search_method, self.shap_func, self.sub_search_method, self.is_reliable,
                                   self.log_queue))
            self.process.start()
            # 设置计时器检查队列
            self.dic_timer = self.startTimer(100)
        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("还未设置ROI区域！")
            msgBox.setWindowTitle("警告")
            msgBox.setWindowIcon(QIcon(":/image/ui_image/警告.png"))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()

    def timerEvent(self, event):
        # 检查队列是否有新的结果
        if not self.dic_queue.empty():
            self.dic_is_ok = True
            self.dic_list_img = self.dic_queue.get()
            self.killTimer(self.dic_timer)
            # 确保进程已经结束
            self.process.join()
            self.cor_listener_thread.stop()

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("DIC算法匹配完成！")
            msgBox.setWindowTitle("恭喜")
            msgBox.setWindowIcon(QIcon(":/image/ui_image/祝贺.png"))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()
            self.logger2.info('DIC算法全部匹配完成')

    # 对后处理页面tab3进行信号处理
    def do_pushButton_result_input_path_clicked(self):
        fileDialog = QFileDialog(self)
        fileDialog.setWindowIcon(QIcon(":/image/ui_image/选择目录.png"))
        self.result_input_path = fileDialog.getExistingDirectory(self, '请选择需要导入的结果文件目录', '')
        self.logger3.info(f'导入结果文件路径为：{self.result_input_path}')

    def do_pushButton_input_img_path_clicked(self):
        fileDialog = QFileDialog(self)
        fileDialog.setWindowIcon(QIcon(':/image/ui_image/选择目录.png'))
        self.img_input_path = fileDialog.getExistingDirectory(self, '请选择需要导入的图像文件目录', '')
        self.logger3.info(f'导入图像文件路径为：{self.img_input_path}')

    def do_pushButton_post_clear_clicked(self):
        self.result_input_path = ''
        self.img_input_path = ''
        self.ui.comboBox_strain_type.setCurrentIndex(0)
        self.ui.comboBox_strain_com_method.setCurrentIndex(0)
        self.ui.radioButton_is_save_result.setChecked(False)
        self.logger3.info('后处理参数已清除，请重新设置')

    def do_pushButton_post_ok_clicked(self):
        if (not self.result_input_path or not self.img_input_path) and not self.cor_is_ok:
            self.logger3.info('未导入结果或文件路径，且未设置完前述参数！')
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("未导入结果或文件路径，且未设置完前述参数！")
            msgBox.setWindowTitle("警告")
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()
            return
        self.is_saved = self.ui.radioButton_is_save_result.isChecked()
        self.strain_type = self.ui.comboBox_strain_type.currentText()
        self.strain_com_method = self.ui.comboBox_strain_com_method.currentText()
        self.logger3.info(f'需要计算应变类型为：{self.strain_type}')
        self.logger3.info(f'采取应变场计算方法为：{self.strain_com_method}')
        self.logger3.info(f'保存结果文件{self.is_saved}，在指定目录：{self.result_save_path}')

        if not self.dic_is_ok:
            self.logger3.info("抱歉，前面的匹配过程还未进行完毕，稍等！")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("抱歉，前面的匹配过程还未进行完毕，稍等！")
            msgBox.setWindowTitle("抱歉")
            msgBox.setWindowIcon(QIcon(':/image/ui_image/信息.png'))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()
        else:
            # 开始进行全局应变场计算
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("开始进行全局应变场计算！")
            msgBox.setWindowTitle("恭喜")
            msgBox.setWindowIcon(QIcon(':/image/ui_image/祝贺.png'))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()
            self.cor_points_list = [dic_img.cor_points for dic_img in self.dic_list_img]
            self.displacement_list = [dic_img.field_disp for dic_img in self.dic_list_img]

            self.post_listener_thread.start()
            self.dic_list_img = dic_algorithm.post_strain_compute(self.img_list, self.roi_top_left_point,
                                                                  self.roi_bottom_right_point, self.cor_points_list,
                                                                  self.displacement_list, self.subset_size, self.step,
                                                                  self.strain_type, self.strain_com_method,
                                                                  self.is_saved,
                                                                  self.result_save_path, log_queue=self.log_queue)
            self.post_listener_thread.stop()
            self.strain_is_ok = True
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("全局应变场计算完成！")
            msgBox.setWindowTitle("恭喜")
            msgBox.setWindowIcon(QIcon(':/image/ui_image/祝贺.png'))
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            msgBox.exec_()
            self.logger3.info('全局应变场计算完成')

        # 绘制初始图像，形成self.figure等变量
        # 清除之前图像上的线条等
        for item in self.pixmap_item.childItems():
            if isinstance(item, (QGraphicsLineItem, QGraphicsRectItem)):
                self.scene.removeItem(item)

        dic_img = self.dic_list_img[-1]  # 匹配的最后一张图像
        self._disp_x_grid = dic_img.field_disp[:, 0].reshape(dic_img.points_num_y, dic_img.points_num_x)
        self._disp_y_grid = dic_img.field_disp[:, 1].reshape(dic_img.points_num_y, dic_img.points_num_x)
        self._strain_xx = dic_img.field_strain[:, :, 0]
        self._strain_yy = dic_img.field_strain[:, :, 1]
        self._strain_xy = dic_img.field_strain[:, :, 2]

        plt.ioff()  # 取消交互式绘图，防止闪现
        dpi = 100
        self.fig, self.ax = plt.subplots(figsize=np.array(dic_img.ref_img.shape) // dpi)
        self.fig.dpi = dpi
        self.ax.set_xlabel('像素x坐标')
        self.ax.set_ylabel('像素y坐标')
        x_tick = np.arange(dic_img.xmin, dic_img.xmax, dic_img.step * 3)
        y_tick = np.arange(dic_img.ymin, dic_img.ymax, dic_img.step * 3)
        self.ax.set_xticks(np.linspace(0, dic_img.points_num_x, len(x_tick)), x_tick)
        self.ax.set_yticks(np.linspace(0, dic_img.points_num_y, len(y_tick)), y_tick)
        self.ax.set_title('x位移阴影图')
        cax = self.ax.imshow(self._disp_x_grid, cmap='hot', interpolation='bicubic')
        self._color_bar = self.fig.colorbar(cax)

        # 将matplotlib图像转换为QPixmap
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        # 得到的buf是一个原始的RGBA缓冲区，我们需要将它转换为QImage
        qimage = QImage(buf, buf.shape[1], buf.shape[0], QImage.Format_RGBA8888)
        self._post_pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self._post_pixmap_item.setFlag(QGraphicsPixmapItem.ItemIsMovable)
        self._post_pixmap_item.setPos(self.pixmap.width(),0)
        self.scene.addItem(self._post_pixmap_item)

    def do_pushButton_save_img_clicked(self):
        try:
            # 保存到文件
            filename = f'{self.result_save_path}/{self.ui.comboBox_img_type.currentText()}_{self.ui.comboBox_img_color_type.currentText()}.png'
            self.fig.savefig(filename,bbox_inches='tight')
            self.logger3.info(f'您已经保存图像到：\n{self.result_save_path}/{self.ui.comboBox_img_type.currentText()}_'
                              f'{self.ui.comboBox_img_color_type.currentText()}.png')
        except Exception as e:
            self.logger3.info(e)

    def do_comboBox_img_currentIndexChanged(self, index):
        if self.ui.comboBox_img_type.currentIndex() >= 3:
            if self.ui.comboBox_img_color_type.count() == 2:
                self.ui.comboBox_img_color_type.removeItem(1)
        else:
            if self.ui.comboBox_img_color_type.count() == 1:
                self.ui.comboBox_img_color_type.addItem('矢量图')
        if not self.strain_is_ok:
            return
        
        try:
            self.scene.removeItem(self._post_pixmap_item)
            self._color_bar.remove()
            self.ax.clear()
        except Exception as e:
            self.logger3.error(e)

        dic_img = self.dic_list_img[-1]
        if self.ui.comboBox_img_color_type.currentText() == '矢量图':  # 绘制矢量图
            if self.ui.comboBox_img_type.currentIndex() == 0:
                quiver = self.ax.quiver(dic_img.grid_x,dic_img.grid_y,self._disp_x_grid,0,color='red',
                                        scale=5,cmap=cm.jet)
                self.ax.set_title('x位移矢量图')
            elif self.ui.comboBox_img_type.currentIndex() == 1:
                quiver = self.ax.quiver(dic_img.grid_x, dic_img.grid_y, 0, self._disp_y_grid, color='red'
                                        ,scale=5,cmap=cm.jet)
                self.ax.set_title('y位移矢量图')
            elif self.ui.comboBox_img_type.currentIndex() == 2:
                quiver = self.ax.quiver(dic_img.grid_x, dic_img.grid_y, self._disp_x_grid, self._disp_x_grid,
                                        color='red',scale=2)
                self.ax.set_title('平面位移矢量图')
            else:
                raise ValueError('图像类型选取错误！')
            self.ax.scatter(dic_img.grid_x,dic_img.grid_y,color='blue',s=1)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        else:  # 绘制阴影图
            if self.ui.comboBox_img_type.currentIndex() == 0:
                self.ax.set_title('x位移阴影图')
                cax = self.ax.imshow(self._disp_x_grid, cmap='hot', interpolation='bicubic')
            elif self.ui.comboBox_img_type.currentIndex() == 1:  # y方向位移
                self.ax.set_title('y位移阴影图')
                cax = self.ax.imshow(self._disp_y_grid,cmap='hot', interpolation='bicubic')
            elif self.ui.comboBox_img_type.currentIndex() == 2:  # xy方向位移
                self.ax.set_title('平面位移阴影图')
                cax = self.ax.imshow(np.sqrt(self._disp_x_grid**2+self._disp_y_grid**2),cmap='hot',
                                     interpolation='bicubic')
            elif self.ui.comboBox_img_type.currentIndex() == 3:  # x方向应变
                self.ax.set_title('x应变阴影图')
                cax = self.ax.imshow(self._strain_xx,cmap='hot',interpolation='bicubic')
            elif self.ui.comboBox_img_type.currentIndex() == 4:  # y方向应变
                self.ax.set_title('y应变阴影图')
                cax = self.ax.imshow(self._strain_yy, cmap='hot', interpolation='bicubic')
            else:  # 剪应变
                self.ax.set_title('xy剪应变阴影图')
                cax = self.ax.imshow(self._strain_xy, cmap='hot', interpolation='bicubic')

            self.ax.set_xlabel('像素x坐标')
            self.ax.set_ylabel('像素y坐标')
            x_tick = np.arange(dic_img.xmin, dic_img.xmax, dic_img.step * 4)
            y_tick = np.arange(dic_img.ymin, dic_img.ymax, dic_img.step * 4)
            self.ax.set_xticks(np.linspace(0, dic_img.points_num_x, len(x_tick)), x_tick)
            self.ax.set_yticks(np.linspace(0, dic_img.points_num_y, len(y_tick)), y_tick)
            self._color_bar = self.fig.colorbar(cax)

        # 将matplotlib图像转换为QPixmap
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        # 得到的buf是一个原始的RGBA缓冲区，我们需要将它转换为QImage
        qimage = QImage(buf, buf.shape[1], buf.shape[0], QImage.Format_RGBA8888)
        self._post_pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self._post_pixmap_item.setFlag(QGraphicsPixmapItem.ItemIsMovable)
        self._post_pixmap_item.setPos(self.pixmap.width(),0)
        self.scene.addItem(self._post_pixmap_item)

        self.logger3.info('绘制完毕')


def main():
    import sys
    # 创建应用程序实例
    app = QtWidgets.QApplication(sys.argv)
    # 创建MyWin窗口实例
    win = MyWin()
    # 显示窗口
    win.show()
    # 运行应用程序
    app.exec_()


if __name__ == '__main__':
    main()

    # 优化点：曲面拟合方法等极其不准；
    # 箭头和热力图太丑了，必须改善；全局位移场推导那块计算点数量对不上好像，得检验一遍
