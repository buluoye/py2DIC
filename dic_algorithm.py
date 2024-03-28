#!/usr/bin/python3.7.4
# -*- coding:utf-8 -*-
# @Author: 叶不落
# @Time: 2024/3/8 10:08
# @desc:DIC主体类的实现

import numpy as np
import time
import re
import csv
import logging
import cv2
import os
from typing import List, Tuple
from scipy.optimize import curve_fit, leastsq
from scipy.linalg import lstsq


# 读取指定目录下的图片或视频，返回images列表
def read_images_from_folder(folder_path: str) -> List[np.ndarray]:
    # 辅助函数，用于进行自然排序
    def natural_sort_key(s: str):
        # 将文本和数字分开并转换数字为整数以供排序
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    images: List[np.ndarray] = []

    # 按自然排序顺序获取所有文件名
    sorted_filenames = sorted(os.listdir(folder_path), key=natural_sort_key)

    for filename in sorted_filenames:
        file = os.path.join(folder_path, filename)
        file = np.fromfile(file, dtype=np.uint8)  # 读取中文路径
        img = cv2.imdecode(file, 0)
        if img is not None:
            images.append(img)

    return images


def read_video_from_file(video_path: str, interval: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)

    images = []
    while cap.isOpened():
        for i in range(interval):
            cap.read()  # 跳过间隔帧

        ret, frame = cap.read()
        if ret:
            images.append(frame)

        if not ret:
            break

    cap.release()
    return images


# 图像前处理，使用灰度消差算法和高斯滤波算法，对图像进行噪声消除，平滑处理
def sobel_filter(img: np.ndarray) -> np.ndarray:
    # 图像灰度消差，返回消差后的新图像
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = (sobel / sobel.max()) * 255
    sobel = sobel.astype(np.uint8)

    return sobel


def gaussian_blur(img: np.ndarray, kernel_size: int = 5):
    # 高斯滤波；高斯滤波核大小默认选取5
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


class QueueHandler(logging.Handler):
    """
    把在该算法中输出的日志都传给队列log_queue；以便在另一个文件中调用该日志信息
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


class DIC:
    # 类属性，各实例对象共享数据，以防重复计算
    cal_points_X: np.ndarray = None  # shape为（n）
    cal_points_Y: np.ndarray = None  # # shape为（m）
    points_num_x: int = None  # 计算点个数
    points_num_y: int = None
    points_num: int = None
    cal_points: np.ndarray = None  # 综合计算点坐标;# shape为（m*n，2），每一行都是一个点的x，y坐标
    grid_x = None  # 计算点网格的x坐标，形状为（m,n)，每点元素为对应索引下点的x坐标
    grid_y = None  # # 计算点网格的y坐标

    init_point: np.ndarray = None  # 可靠性传导中需要的第一个初始种子点，一般取中心点
    index_init_point: np.ndarray = None  # 初始种子点的索引
    # ROI感兴趣区域左上角和右下角点坐标
    xmin = None
    xmax = None
    ymin = None
    ymax = None

    subset_size = None
    half_subset_size = None
    step = None
    search_subset_size = None  # 搜索区域限制，提高搜索效率
    half_search_subset_size = None

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('DIC算法细节')
    fileHandler = logging.FileHandler('日志记录.log')  # 汇总文件日志记录
    logger.addHandler(fileHandler)

    def __init__(self, ref_img: np.ndarray, tar_img: np.ndarray):
        # 导入外部数据
        self.ref_img = ref_img
        self.tar_img = tar_img
        self.ref_img_doi = ref_img[DIC.xmin:DIC.xmax + 1, DIC.ymin:DIC.ymax + 1]
        self.tar_img_doi = tar_img[DIC.xmin:DIC.xmax + 1, DIC.ymin:DIC.ymax + 1]

        # 计算后更新存储信息数据
        self.cor_points: np.ndarray = None  # 匹配点的坐标
        self.field_disp: np.ndarray = None  # 全局位移场每个点的相对位移，# shape为（m*n，2），每一行都是一个点的x，y位移
        self.disp_grid = None  # 全局位移场，，shape为（m,n,2)
        self.field_strain = None  # 全局应变场，# shape为（m，n，3），分别代表x方向线应变，y方向线应变和切应变

        self.ref_img_sobelx = cv2.Sobel(self.ref_img, cv2.CV_64F, 1, 0, ksize=5)  # 参考图像的灰度梯度
        self.ref_img_sobely = cv2.Sobel(self.ref_img, cv2.CV_64F, 0, 1, ksize=5)

        # 常用计算数据
        self.ref_sizeX = np.size(self.ref_img, 0)  # 感兴趣区域大小
        self.ref_sizeY = np.size(self.ref_img, 1)
        self.tar_sizeX = np.size(self.tar_img, 0)
        self.tar_sizeY = np.size(self.tar_img, 1)
        self.max_iter = 20  # 最大迭代次数
        self.iter_thre = 1e-3  # 迭代法中的阈值

    @classmethod
    def get_calculate_points(cls, pt1: np.ndarray, pt2: np.ndarray, subset_size: int, step: int, search_size=100,
                             log_queue=None):
        """
        根据两个对角点坐标p1和p2按照计算步长得到所有计算点坐标，自动更新self，无返回
        """
        points = []
        x1 = int(min(pt1[0], pt2[0]))
        x2 = int(max(pt1[0], pt2[0]))
        y1 = int(min(pt1[1], pt2[1]))
        y2 = int(max(pt1[1], pt2[1]))
        cls.xmin, cls.xmax, cls.ymin, cls.ymax = x1, x2, y1, y2

        cls.subset_size = subset_size
        cls.half_subset_size = cls.subset_size // 2
        cls.step = step
        cls.search_subset_size = search_size
        cls.half_search_subset_size = search_size // 2

        cls.cal_points_X = np.arange(x1 + cls.half_subset_size, x2 - cls.half_subset_size, cls.step).flatten()
        cls.cal_points_Y = np.arange(y1 + cls.half_subset_size, y2 - cls.half_subset_size, cls.step).flatten()
        cls.points_num_x = len(cls.cal_points_X)
        cls.points_num_y = len(cls.cal_points_Y)
        cls.points_num = cls.points_num_x * cls.points_num_y

        cls.grid_x, cls.grid_y = np.meshgrid(cls.cal_points_X, cls.cal_points_Y)
        # 其是一个n行两列数组，n为计算点总个数，每一行都是一个点的x，y坐标；计算点顺序从左到右，从上到下
        cls.cal_points = np.concatenate(
            (cls.grid_x.reshape(-1, 1), cls.grid_y.reshape(-1, 1)), axis=1)

        # 初始种子点的坐标和索引
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        dist = np.sqrt(np.square(cls.grid_x.reshape(-1, 1) - x) + np.square(cls.grid_y.reshape(-1, 1) - y))
        cls.index_init_point = np.argmin(dist)
        a = cls.grid_x.reshape(-1, 1)[cls.index_init_point]
        cls.init_point = np.concatenate((cls.grid_x.reshape(-1, 1)[cls.index_init_point],
                                         cls.grid_y.reshape(-1, 1)[cls.index_init_point]))
        if log_queue:
            q_handler = QueueHandler(log_queue)
            cls.logger.addHandler(q_handler)

    def get_tar_subset(self, cal_point_x: int, cal_point_y: int) -> np.ndarray:
        """
        对于匹配图像，给定一个计算点，以其为中心，得到对应的计算子区
        :param cal_point_x:计算中心点的x坐标
        :param cal_point_y:计算中心点的y坐标
        :return:返回计算子区存储的灰度信息矩阵
        """
        current_gSubset = self.tar_img[cal_point_y - DIC.half_subset_size:cal_point_y + DIC.half_subset_size + 1,
                                       cal_point_x - DIC.half_subset_size:cal_point_x + DIC.half_subset_size + 1]
        return current_gSubset

    def is_subset_inside(self, point_x, point_y):
        """判断以该点为中心的子区是否在完整大图像中"""
        return ((0 + DIC.half_subset_size <= point_x < self.ref_sizeX - DIC.half_subset_size) and
                (0 + DIC.half_subset_size <= point_y < self.ref_sizeY - DIC.half_subset_size))

    def to_grid_displacement(self):
        try:
            disp_x_grid = self.field_disp[:, 0].reshape(DIC.points_num_y,DIC.points_num_x)
            disp_y_grid = self.field_disp[:, 1].reshape(DIC.points_num_y,DIC.points_num_x)
            self.disp_grid = np.dstack((disp_x_grid, disp_y_grid))
        except:
            pass

    # 各种相关准则实现
    @staticmethod
    def all_criterion(ref_template, tar_template, criterion: str) -> Tuple[float, bool]:
        # 综合各种相关准则，总函数调用，返回匹配分数值以及最佳是最大值or最小值
        def CCOEFF_NORMED_criterion(template1: np.ndarray, template2: np.ndarray) -> float:
            # 零均值归一化互相关标准，返回两个区域之间的匹配分数；介于-1（完全不相关）和1（完全相关）之间
            result = cv2.matchTemplate(template1, template2, cv2.TM_CCOEFF_NORMED)
            # 获取最大匹配值
            _, max_val, _, _ = cv2.minMaxLoc(result)

            return max_val

        def CCOEFF_criterion(template1, template2):
            result = cv2.matchTemplate(template1, template2, cv2.TM_CCOEFF)
            # 获取最大匹配值
            _, max_val, _, _ = cv2.minMaxLoc(result)

            return max_val

        def SQDIFF_criterion(template1, template2, ):
            result = cv2.matchTemplate(template1, template2, cv2.TM_SQDIFF)
            # 获取最小匹配值
            min_val, _, _, _ = cv2.minMaxLoc(result)

            return min_val

        def SQDIFF_NORMED_criterion(template1, template2):
            result = cv2.matchTemplate(template1, template2, cv2.TM_SQDIFF_NORMED)
            # 获取最小匹配值
            min_val, _, _, _ = cv2.minMaxLoc(result)

            return min_val

        def CCORR_criterion(template1, template2):
            result = cv2.matchTemplate(template1, template2, cv2.TM_CCORR)
            # 获取最大匹配值
            _, max_val, _, _ = cv2.minMaxLoc(result)

            return max_val

        def CCORR_NORMED_criterion(template1, template2):
            result = cv2.matchTemplate(template1, template2, cv2.TM_CCORR_NORMED)
            # 获取最大匹配值
            _, max_val, _, _ = cv2.minMaxLoc(result)

            return max_val

        assert ref_template.shape == tar_template.shape, "两个模板形状不匹配"
        max_or_min = True  # True表示分数越高，匹配程度越高
        if criterion == "CCOEFF_NORMED准则":
            return CCOEFF_NORMED_criterion(ref_template, tar_template), max_or_min
        elif criterion == "CCOEFF准则":
            return CCOEFF_criterion(ref_template, tar_template), max_or_min
        elif criterion == "CCORR_NORMED准则":
            return CCORR_NORMED_criterion(ref_template, tar_template), max_or_min
        elif criterion == "CCORR准则":
            return CCORR_criterion(ref_template, tar_template), max_or_min
        elif criterion == "SQDIFF_NORMED准则":
            max_or_min = False
            return SQDIFF_NORMED_criterion(ref_template, tar_template), max_or_min
        elif criterion == "SQDIFF准则":
            max_or_min = False
            return SQDIFF_criterion(ref_template, tar_template), max_or_min
        else:
            raise ValueError("相关函数选取错误！")

    def integer_pixel_search(self, point: np.ndarray, method: str = "逐点搜索",
                             cor_criterion: str = "CCOEFF_NORMED准则") -> np.ndarray:
        """
        整数像素搜索方法，给定特定点，需要使用的搜索方法以及相关准则，返回最佳匹配点坐标
        """

        # 各种搜索算法细节实现
        def point_point_search(template, criterion: str = cor_criterion) -> np.ndarray:
            """针对给定点，根据指定的相关准则，通过逐点遍历，找到匹配程度最高的点，返回其坐标"""
            if criterion == "CCORR_NORMED准则":  # 归一化相关准则
                cor_score = cv2.matchTemplate(self.tar_img, template, method=cv2.TM_CCORR_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(cor_score)
                cor_point_x = max_loc[0] + DIC.half_subset_size
                cor_point_y = max_loc[1] + DIC.half_subset_size
                return np.array((cor_point_x, cor_point_y))
            elif criterion == "SQDIFF准则":  # 平方差准则
                cor_score = cv2.matchTemplate(self.tar_img, template, method=cv2.TM_SQDIFF)
                min_val, _, min_loc, _ = cv2.minMaxLoc(cor_score)
                cor_point_x = min_loc[0] + DIC.half_subset_size
                cor_point_y = min_loc[1] + DIC.half_subset_size
                return np.array((cor_point_x, cor_point_y))
            elif criterion == "SQDIFF_NORMED准则":  # 平方差归一化准则
                cor_score = cv2.matchTemplate(self.tar_img, template, method=cv2.TM_SQDIFF_NORMED)
                min_val, _, min_loc, _ = cv2.minMaxLoc(cor_score)
                cor_point_x = min_loc[0] + DIC.half_subset_size
                cor_point_y = min_loc[1] + DIC.half_subset_size
                return np.array((cor_point_x, cor_point_y))
            elif criterion == "CCORR准则":  # 相关准则
                cor_score = cv2.matchTemplate(self.tar_img, template, method=cv2.TM_CCORR)
                _, max_val, _, max_loc = cv2.minMaxLoc(cor_score)
                cor_point_x = max_loc[0] + DIC.half_subset_size
                cor_point_y = max_loc[1] + DIC.half_subset_size
                return np.array((cor_point_x, cor_point_y))
            elif criterion == "CCOEFF准则":  # 零均值相关准则
                cor_score = cv2.matchTemplate(self.tar_img, template, method=cv2.TM_CCOEFF)
                _, max_val, _, max_loc = cv2.minMaxLoc(cor_score)
                cor_point_x = max_loc[0] + DIC.half_subset_size
                cor_point_y = max_loc[1] + DIC.half_subset_size
                return np.array((cor_point_x, cor_point_y))
            elif criterion == "CCOEFF_NORMED准则":  # 零均值归一化相关准则
                cor_score = cv2.matchTemplate(self.tar_img, template, method=cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(cor_score)
                cor_point_x = max_loc[0] + DIC.half_subset_size
                cor_point_y = max_loc[1] + DIC.half_subset_size
                return np.array((cor_point_x, cor_point_y))
            else:
                raise ValueError('相关函数选取错误！')

        def normal_point_point_search(template, criterion: str = cor_criterion) -> np.ndarray:
            """自己实现的逐点搜索，循环遍历，没有使用cv2的matchTemplate函数，与后面的粗细搜索做对比"""
            best_value = np.inf
            for x in range(0 + DIC.half_subset_size, self.ref_sizeX - DIC.half_subset_size, ):
                for y in range(0 + DIC.half_subset_size, self.ref_sizeY - DIC.half_subset_size):
                    current_gSubset = self.get_tar_subset(x, y)
                    match_value, max_or_min = self.all_criterion(template, current_gSubset, criterion)
                    if max_or_min:
                        if match_value > best_value:
                            best_match = np.array((x, y))
                            best_value = match_value
                    else:
                        if match_value < best_value:
                            best_match = np.array((x, y))
                            best_value = match_value
            return best_match

        def cu_xi_search(template, criterion: str = cor_criterion) -> np.ndarray:
            """
            分两步，步长分别为3像素、1像素;
            存在问题：粗搜索中匹配度最大值所处的区域很可能并不是最佳匹配点所处区域，
            所以需要取匹配度前20的点，各自进行细化搜索，得到最佳点
            """
            matches = []  # 匹配点坐标及匹配值
            for x in range(0 + DIC.half_subset_size, self.ref_sizeX - DIC.half_subset_size, 3):
                for y in range(0 + DIC.half_subset_size, self.ref_sizeY - DIC.half_subset_size, 3):
                    current_gSubset = self.get_tar_subset(x, y)
                    match_value, max_or_min = self.all_criterion(template, current_gSubset, criterion)
                    matches.append((match_value, (x, y)))
            if max_or_min:
                matches.sort(key=lambda match: match[0], reverse=True)
            else:
                matches.sort(key=lambda match: match[0])
            top_matches = matches[:20]

            refined_matches = []
            for value, (x1, y1) in top_matches:
                for x_offset in range(-2, 3):
                    for y_offset in range(-2, 3):
                        # 对每个点和其周围的位移点进行搜索
                        refined_x, refined_y = x1 + x_offset, y1 + y_offset
                        # 确保搜索的点在图像范围内
                        if self.is_subset_inside(refined_x, refined_y):
                            current_gSubset = self.get_tar_subset(refined_x, refined_y)
                            refined_value, _ = self.all_criterion(template, current_gSubset, criterion)
                            refined_matches.append((refined_value, (refined_x, refined_y)))
            if max_or_min:
                refined_matches.sort(key=lambda match: match[0], reverse=True)
            else:
                refined_matches.sort(key=lambda match: match[0])

            best_match = np.array(refined_matches[0][1])
            return best_match

        def shizi_search(template, criterion: str = cor_criterion) -> np.ndarray:
            best_cor_score = np.inf
            cor_point_x = ref_point[0]  # 外部中存在ref_point变量，直接使用
            cor_point_y = ref_point[1]
            old_x = 0
            old_y = 0
            while True:
                X = range(0 + DIC.half_subset_size, self.ref_sizeX - DIC.half_subset_size)
                for x in X:
                    current_gSubset = self.get_tar_subset(x, point[1])
                    cor_score, max_or_min = self.all_criterion(template, current_gSubset, criterion)
                    if max_or_min:
                        if cor_score > best_cor_score:
                            best_cor_score = cor_score
                            cor_point_x = x
                    else:
                        if cor_score < best_cor_score:
                            best_cor_score = cor_score
                            cor_point_x = x

                Y = range(0 + DIC.half_subset_size, self.ref_sizeY - DIC.half_subset_size)
                for y in Y:
                    current_gSubset = self.get_tar_subset(cor_point_x, y)
                    cor_score, max_or_min = self.all_criterion(template, current_gSubset, criterion)
                    if max_or_min:
                        if cor_score > best_cor_score:
                            best_cor_score = cor_score
                            cor_point_y = y
                    else:
                        if cor_score < best_cor_score:
                            best_cor_score = cor_score
                            cor_point_y = y
                if old_x == cor_point_x and old_y == cor_point_y:
                    break
                old_x = cor_point_x
                old_y = cor_point_y

            return np.array((cor_point_x, cor_point_y))

        def cuxi_shizi_search(template, criterion: str = cor_criterion) -> np.ndarray:
            # 先进行步长为4的粗搜索，再在x方向和y方向进行较小范围（7像素）的十字搜索
            matches = []  # 匹配点坐标及匹配值
            for x in range(0 + DIC.half_subset_size, self.ref_sizeX - DIC.half_subset_size, 3):
                for y in range(0 + DIC.half_subset_size, self.ref_sizeY - DIC.half_subset_size, 3):
                    current_gSubset = self.get_tar_subset(x, y)
                    match_value, max_or_min = self.all_criterion(template, current_gSubset, criterion)
                    matches.append((match_value, (x, y)))
            if max_or_min:
                matches.sort(key=lambda match: match[0], reverse=True)
            else:
                matches.sort(key=lambda match: match[0])
            top_matches = matches[:20]
            best_cross_matches = []

            for value, (x, y) in top_matches:
                # 在X方向上的细化搜索 (假设横向搜索范围是max_offset)
                best_match_x, best_value_x = x, value
                for dx in range(-3, 4):
                    search_x = x + dx
                    if self.is_subset_inside(search_x, y):  # 确保细化搜索不会超出图像范围
                        current_gSubset = self.get_tar_subset(search_x, y)
                        match_value, _ = self.all_criterion(template, current_gSubset, criterion)
                        if max_or_min:
                            if match_value > best_value_x:
                                best_match_x, best_value_x = search_x, match_value
                        else:
                            if match_value < best_value_x:
                                best_match_x, best_value_x = search_x, match_value
                # 固定X坐标，在Y方向上的细化搜索
                best_match_y, best_value_y = y, best_value_x
                for dy in range(-3, 4):
                    search_y = y + dy
                    if self.is_subset_inside(best_match_x, search_y):  # 确保细化搜索不会超出图像范围
                        current_gSubset = self.get_tar_subset(best_match_x, search_y)
                        match_value, _ = self.all_criterion(template, current_gSubset, criterion)
                        if max_or_min:
                            if match_value > best_value_y:
                                best_match_y, best_value_y = search_y, match_value
                        else:
                            if match_value < best_value_y:
                                best_match_y, best_value_y = search_y, match_value
                # 保存十字搜索得到的匹配点和匹配值
                best_cross_matches.append((best_value_y, (best_match_x, best_match_y)))

            # 对细化搜索得到的匹配点进行排序并选择最佳的匹配点
            if max_or_min:
                best_cross_matches.sort(key=lambda match: match[0], reverse=True)
            else:
                best_cross_matches.sort(key=lambda match: match[0])
            best_match = np.array(best_cross_matches[0][1])

            return best_match

        def GA_search(template, criterion: str = cor_criterion) -> np.ndarray:
            pass

        def PSO_search():
            pass

        ref_point = point.astype('int64')
        ref_template = self.ref_img[ref_point[1] - DIC.half_subset_size:ref_point[1] + DIC.half_subset_size + 1,
                       ref_point[0] - DIC.half_subset_size:ref_point[0] + DIC.half_subset_size + 1]
        if method == '逐点搜索':
            return point_point_search(ref_template, criterion=cor_criterion)
        elif method == '普通逐点搜索':
            return normal_point_point_search(ref_template, criterion=cor_criterion)
        elif method == '粗细搜索':
            return cu_xi_search(ref_template, criterion=cor_criterion)
        elif method == '粗细十字搜索':
            return cuxi_shizi_search(ref_template, criterion=cor_criterion)
        elif method == '十字搜索':
            return shizi_search(ref_template, criterion=cor_criterion)
        elif method == "基于遗传算法搜索":
            return GA_search(ref_template, criterion=cor_criterion)
        else:
            raise ValueError("整数像素搜索算法选取错误！")

    @staticmethod
    def gray_interpolation_algorithm(img: np.ndarray, factor: int = 5,
                                     inter_algorithm: str = "双线性插值") -> np.ndarray:
        """
        将整数像素级别的图像采用不同插值算法插值成亚像素级别的图像
        :param img: 原图像
        :param factor: 需要插值的倍数
        :param inter_algorithm: 采用插值的算法
        :return: 返回插值后的图像
        """
        # 计算新图像的尺寸
        new_width = int(img.shape[1] * factor)
        new_height = int(img.shape[0] * factor)

        if inter_algorithm == '最近邻插值':
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        elif inter_algorithm == '双线性插值':
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        elif inter_algorithm == '双三次B样条插值':
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError("Unsupported interpolation method")

    def sub_pixel_search(self, ref_point: np.ndarray, cor_point: np.ndarray, method: str = "曲面拟合法",
                         shape_fuc="一阶形函数", cor_criterion: str = "零均值归一化互相关标准",
                         inter_method: str = "双线性插值") -> tuple[float, np.ndarray]:
        """
        :param ref_point:参考点
        :param cor_point: 整数像素搜索得到的近似匹配点（整数精度级别）
        :param method: 选择亚像素搜索方法，默认为曲面拟合法
        :param shape_fuc: 形函数选取
        :param cor_criterion: 选择相关准则，默认零均值归一化互相关标准
        :param inter_method:灰度插值算法
        :return: 返回匹配值（可靠性值，越高越匹配，越可靠）以及匹配到的亚像素精度级别的点坐标，一维数组
        """

        def quadric_surface_fitting_method() -> tuple[float, np.ndarray]:
            def quadratic_surface(data, a1, a2, a3, a4, a5, a6):
                # 二次曲面拟合模型，六个参数
                x, y = data
                return a1 * x ** 2 + a2 * y ** 2 + a3 * x * y + a4 * x + a5 * y + a6

            # 提取初始匹配子区
            x = cor_point[0]
            y = cor_point[1]
            ref_subset = self.ref_img[y - DIC.half_subset_size: y + DIC.half_subset_size + 1,
                                      x - DIC.half_subset_size: x + DIC.half_subset_size + 1]

            # 初始化存储相关函数值的列表
            correlation_values = []
            # 遍历周围5×5区域的像素点
            for i in range(x - 1, x + 2):
                for j in range(y - 1, y + 2):
                    # 提取目标子区
                    if not self.is_subset_inside(i, j):
                        correlation_values.append(np.nan)
                        continue
                    tar_subset = self.get_tar_subset(i, j)
                    # 计算相关函数值
                    correlation_value, _ = self.all_criterion(ref_subset, tar_subset, cor_criterion)
                    correlation_values.append(correlation_value)
            # 构建初始值
            guess = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            x = np.arange(x - 1, x + 2)
            y = np.arange(y - 1, y + 2)
            X, Y = np.meshgrid(x, y)
            try:
                popt, pcov = curve_fit(f=quadratic_surface, xdata=(X.ravel(), Y.ravel()),ydata=correlation_values,
                                       method='trf',maxfev=1000, p0=guess)
            except Exception as e:
                print(e)
                return np.inf, cor_point

            # 寻找极值点坐标
            a, b, c, d, e, f = popt
            A = np.array([[2 * a, c], [c, 2 * b]])
            B = np.array([-d, -e])
            # 求解线性方程A*[x, y] = B
            sub_cor_point = np.round(np.linalg.solve(A, B), decimals=3)

            # 得到其极值点对应的相关函数值，可以用作可靠性分析判断
            best_cor_value = quadratic_surface(sub_cor_point, *popt)
            if cor_criterion in ["SQDIFF_NORMED准则", "SQDIFF准则"]:
                czncc = 1 - 0.5 * best_cor_value
            else:
                czncc = best_cor_value

            return czncc, sub_cor_point

        def gray_gradient_search(factor=5) -> tuple[float, np.ndarray]:
            """
            基于灰度梯度的搜索方法
            :param windowsize: 灰度梯度窗口搜索大小
            :param factor: 插值倍数
            :param ref_point: 参考点坐标
            :param cor_point: 整数级别匹配坐标
            :param inter_method: 插值方法
            :return: 返回亚像素级别匹配点
            """
            # Compute gradients for both reference and target images.
            resized_ref_image = self.gray_interpolation_algorithm(self.ref_img, inter_algorithm=inter_method)
            resized_tar_image = self.gray_interpolation_algorithm(self.tar_img, inter_algorithm=inter_method)

            # grad_x_ref = cv2.Sobel(resized_ref_image, cv2.CV_64F, 1, 0, ksize=5)
            # grad_y_ref = cv2.Sobel(resized_ref_image, cv2.CV_64F, 0, 1, ksize=5)
            grad_x_tar = cv2.Sobel(resized_tar_image, cv2.CV_64F, 1, 0, ksize=5)
            grad_y_tar = cv2.Sobel(resized_tar_image, cv2.CV_64F, 0, 1, ksize=5)

            initial_displacement: np.ndarray = cor_point - ref_point

            # Prepare the A matrix and B vector for least squares problem.
            A = []
            B = []
            for i in range(-DIC.half_subset_size, DIC.half_subset_size + 1):
                for j in range(-DIC.half_subset_size, DIC.half_subset_size + 1):
                    # Coordinates in the subset
                    x_subset = ref_point[0] * factor + i
                    y_subset = ref_point[1] * factor + j

                    # Ignore points that are outside the image boundaries.
                    if x_subset < 0 or y_subset < 0 or x_subset >= resized_ref_image.shape[1] or y_subset >= resized_tar_image.shape[0]:
                        continue

                    # Assume displacement (dx, dy) for the corresponding point in the target image
                    # is very close to the initial displacement.
                    dx_init, dy_init = initial_displacement * factor

                    # Coordinates in the target image, with initial displacement applied.
                    x_tgt = x_subset + dx_init
                    y_tgt = y_subset + dy_init

                    # Ignore points that are outside the image boundaries.
                    if x_tgt < 0 or y_tgt < 0 or x_tgt >= resized_tar_image.shape[1] or x_tgt >= resized_tar_image.shape[0]:
                        continue

                    # Compute gradient values at the current coordinate in the tar image.
                    Ix_tar = grad_x_tar[y_tgt, x_tgt]
                    Iy_tar = grad_y_tar[y_tgt, x_tgt]

                    # Calculate intensity differences between reference and target images.
                    delta_I = resized_ref_image[y_subset, x_subset] - resized_tar_image[y_tgt, x_tgt]

                    # Add to the A matrix and B vector
                    A.append([Ix_tar, Iy_tar])
                    B.append([delta_I])

            # Convert lists to NumPy arrays.
            A = np.array(A)
            B = np.array(B)

            # Use least squares to solve for displacement.
            displacement, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            displacement = displacement.flatten()

            best_match = np.round(cor_point - displacement,3)
            ref_template = self.ref_img[cor_point[1] - DIC.half_subset_size:cor_point[1] + DIC.half_subset_size + 1,
                                        cor_point[0] - DIC.half_subset_size:cor_point[0] + DIC.half_subset_size + 1]
            tar_template = self.get_tar_subset(*np.rint(best_match).astype(int))
            max_or_min,cor_value = self.all_criterion(ref_template,tar_template,cor_criterion)
            if max_or_min:
                czncc = cor_value
            else:
                czncc = 1 - 0.5 * cor_value

            return czncc,best_match

        def newton_raphson_iter() -> tuple[float, np.ndarray]:
            def bicubic_Bspline_interp(img, PcoordInt):
                xInt = np.floor(PcoordInt)
                deltaX = PcoordInt - xInt
                numPt = xInt.shape[1]
                MBT = np.array([[-1, 3, -3, 1],
                                [3, -6, 0, 4],
                                [-3, 3, 3, 1],
                                [1, 0, 0, 0]]) / 6
                deltaMatX = MBT @ np.vstack((deltaX[0] ** 3, deltaX[0] ** 2, deltaX[0], np.ones(numPt)))
                deltaMatY = MBT @ np.vstack((deltaX[1] ** 3, deltaX[1] ** 2, deltaX[1], np.ones(numPt)))

                # 参考图像中计算点的索引
                index = np.tile(np.vstack((xInt[1] - 2, xInt[1] - 1, xInt[1], xInt[1] + 1)), (4, 1)) * len(
                    img) + np.vstack(
                    (np.tile(xInt[0] - 1, (4, 1)), np.tile(xInt[0], (4, 1)), np.tile(xInt[0] + 1, (4, 1)),
                     np.tile(xInt[0] + 2, (4, 1)))) - 1
                D_all = img.flatten('F')[index.astype('int32')]
                tarIntp = np.tile(deltaMatY, (4, 1)) * D_all * np.vstack((np.tile(deltaMatX[0], (4, 1)),
                                                                          np.tile(deltaMatX[1], (4, 1)),
                                                                          np.tile(deltaMatX[2], (4, 1)),
                                                                          np.tile(deltaMatX[3], (4, 1))))

                return np.sum(tarIntp, 0).reshape(-1, 1)

            def p_to_wrap(p1):
                """把整数像素级别的初始形变参数转换成扭曲向量"""
                if len(p1) == 6:
                    return np.array([[1 + p1[1][0], p1[2][0], p1[0][0]],
                                     [p1[4][0], 1 + p1[5][0], p1[3][0]],
                                     [0, 0, 1]])
                else:
                    s1 = 2 * p1[1][0] + p1[1][0] ** 2 + p1[0][0] * p1[3][0]
                    s2 = 2 * p1[0][0] * p1[4][0] + 2 * (1 + p1[1][0]) * p1[2][0]
                    s3 = p1[2][0] ** 2 + p1[0][0] * p1[5][0]
                    s4 = 2 * p1[0][0] * (1 + p1[1][0])
                    s5 = 2 * p1[0][0] * p1[2][0]
                    s6 = p1[0][0] ** 2
                    s7 = 1 / 2 * (p1[6][0] * p1[3][0] + 2 * (1 + p1[1][0]) * p1[7][0] + p1[0][0] * p1[9][0])
                    s8 = p1[2][0] * p1[7][0] + p1[1][0] * p1[8][0] + p1[6][0] * p1[4][0] + p1[0][0] * p1[10][0] + p1[8][
                        0] + p1[1][0]
                    s9 = 1 / 2 * (p1[6][0] * p1[5][0] + 2 * (1 + p1[8][0]) * p1[2][0] + p1[0][0] * p1[11][0])
                    s10 = p1[6][0] + p1[6][0] * p1[1][0] + p1[0][0] * p1[7][0]
                    s11 = p1[0][0] + p1[6][0] * p1[2][0] + p1[0][0] * p1[8][0]
                    s12 = p1[0][0] * p1[6][0]
                    s13 = p1[7][0] ** 2 + p1[6][0] * p1[9][0]
                    s14 = 2 * p1[6][0] * p1[10][0] + 2 * p1[7][0] * (1 + p1[8][0])
                    s15 = 2 * p1[8][0] + p1[8][0] ** 2 + p1[6][0] * p1[11][0]
                    s16 = 2 * p1[6][0] * p1[7][0]
                    s17 = 2 * p1[6][0] * (1 + p1[8][0])
                    s18 = p1[6][0] ** 2

                    return np.array([[1 + s1, s2, s3, s4, s5, s6],
                                     [s7, 1 + s8, s9, s10, s11, s12],
                                     [s13, s14, 1 + s15, s16, s17, s18],
                                     [1 / 2 * p1[3][0], p1[4][0], 1 / 2 * p1[5][0], 1 + p1[1][0], p1[2][0], p1[0][0]],
                                     [1 / 2 * p1[9][0], p1[10][0], 1 / 2 * p1[11][0], p1[7][0], 1 + p1[8][0], p1[6][0]],
                                     [0, 0, 0, 0, 0, 1]]).astype('float64')

            def iter_ICGN1(pCoord: np.ndarray, p1):
                # 单个点迭代，一阶形函数
                Iter = 0
                deltaVecX = np.arange(-DIC.half_subset_size, DIC.half_subset_size + 1)
                deltaVecY = np.arange(-DIC.half_subset_size, DIC.half_subset_size + 1)
                deltax, deltay = np.meshgrid(deltaVecX, deltaVecY, indexing='xy')
                localSubHom = np.concatenate((deltax.reshape(1, -1), deltay.reshape(1, -1),
                                              np.ones((1, self.subset_size * self.subset_size))), axis=0)
                localSub = localSubHom[0:2].T
                gradx_img = self.ref_img_sobelx
                grady_img = self.ref_img_sobely

                localSub = localSub / np.tile(np.ceil([self.subset_size / 2, self.subset_size / 2]), (len(localSub), 1))
                M = np.diag(
                    [1, 1 / self.subset_size, 1 / self.subset_size, 1, 1 / self.subset_size, 1 / self.subset_size])

                # 估计海森矩阵（Hessian）的deltaf
                nablaf_x = gradx_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
                nablaf_y = grady_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
                nablaf = np.concatenate((nablaf_x.reshape((-1, 1), order='F'), nablaf_y.reshape((-1, 1), order='F')), 1)
                J = np.vstack((nablaf[:, 0], localSub[:, 0] * nablaf[:, 0], localSub[:, 1] * nablaf[:, 0], nablaf[:, 1],
                               localSub[:, 0] * nablaf[:, 1], localSub[:, 1] * nablaf[:, 1])).T
                H = J.T @ J
                inv_H = np.linalg.inv(H)
                fSubset = self.ref_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
                fSubset = fSubset.reshape((-1, 1), order='F')

                deltafVec = fSubset - np.mean(fSubset)
                deltaf = np.sqrt(np.sum(deltafVec ** 2))
                inv_H_J = inv_H @ J.T

                # 由参数向量计算扭曲函数
                warp = p_to_wrap(p1)

                thre = 1
                # 迭代优化参数向量p
                while thre > self.iter_thre and Iter < self.max_iter or Iter == 0:
                    # 由扭曲函数求得点在目标图像中的坐标
                    gIntep = warp @ localSubHom
                    PcoordInt = pCoord + gIntep - np.array([[0], [0], [1]])

                    # 所有点仍然位于目标图像内
                    # if np.prod(PcoordInt[0:2].min(1) >= [0, 0]) and np.prod(
                    #         PcoordInt[0:2].min(1) <= [self.ref_sizeX , self.ref_sizeY ]):
                    # 双三次B样条插值
                    tarIntp = bicubic_Bspline_interp(self.tar_img, PcoordInt)

                    deltagVec = tarIntp - np.mean(tarIntp)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))

                    delta = deltafVec - deltaf / deltag * deltagVec

                    deltap = -inv_H_J @ delta
                    deltap = M @ deltap

                    # 更新扭曲函数
                    deltawarp = p_to_wrap(deltap)
                    warp = warp @ np.linalg.inv(deltawarp)

                    thre = np.sqrt(deltap[0] ** 2 + deltap[3] ** 2)
                    Iter = Iter + 1

                    # 更新参数向量
                    p1 = np.array(
                        [warp[0, 2], warp[0, 0] - 1, warp[0, 1], warp[1, 2], warp[1, 0], warp[1, 1] - 1]).reshape(
                        -1, 1)
                    Disp = np.array([warp[0, 2], warp[1, 2]])

                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    Czncc = 1 - 0.5 * Cznssd

                return p1, Czncc, Iter, Disp

            def iter_ICGN2(pCoord: np.ndarray, p1):
                # 单个点迭代,二阶形函数
                Iter = 0
                deltaVecX = np.arange(-DIC.half_subset_size, DIC.half_subset_size + 1)
                deltaVecY = np.arange(-DIC.half_subset_size, DIC.half_subset_size + 1)
                deltax, deltay = np.meshgrid(deltaVecX, deltaVecY, indexing='xy')
                localSubHom = np.concatenate((deltax.reshape(1, -1), deltay.reshape(1, -1),
                                              np.ones((1, self.subset_size * self.subset_size))), axis=0)
                localSub = localSubHom[0:2].T
                gradx_img = self.ref_img_sobelx
                grady_img = self.ref_img_sobely

                localSub = localSub / np.tile(np.ceil([self.subset_size / 2, self.subset_size / 2]), (len(localSub), 1))
                M = np.diag(
                    [1, 1 / self.subset_size, 1 / self.subset_size, 1 / self.subset_size ** 2,
                     1 / self.subset_size ** 2,
                     1 / self.subset_size ** 2,
                     1, 1 / self.subset_size, 1 / self.subset_size, 1 / self.subset_size ** 2,
                     1 / self.subset_size ** 2,
                     1 / self.subset_size ** 2])

                nablaf_x = gradx_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
                nablaf_y = grady_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
                nablaf = np.concatenate((nablaf_x.reshape((-1, 1), order='F'), nablaf_y.reshape((-1, 1), order='F')), 1)
                deltaW2P = np.vstack((np.ones((1, self.subset_size ** 2)), localSub[:, 0], localSub[:, 1],
                                      1 / 2 * localSub[:, 0] ** 2, localSub[:, 0] * localSub[:, 1],
                                      1 / 2 * localSub[:, 1] ** 2)).T
                J = np.hstack((np.tile(nablaf[:, 0].reshape(-1, 1), (1, 6)) * deltaW2P,
                               np.tile(nablaf[:, 1].reshape(-1, 1), (1, 6)) * deltaW2P))
                H = J.T @ J
                inv_H = np.linalg.inv(H)
                fSubset = self.ref_img[np.ix_(pCoord[0] - 1 + deltaVecX, pCoord[1] - 1 + deltaVecY)]
                fSubset = fSubset.reshape((-1, 1), order='F')
                deltafVec = fSubset - np.mean(fSubset)
                deltaf = np.sqrt(np.sum(deltafVec ** 2))
                inv_H_J = inv_H @ J.T
                # 由参数向量计算扭曲函数
                warp = p_to_wrap(p1)
                thre = 1
                # 迭代优化参数向量p
                while thre > self.iter_thre and Iter < self.max_iter or Iter == 0:
                    # 由扭曲函数求得点在目标图像中的坐标
                    gIntep = warp[[3, 4, 5]] @ np.vstack(
                        (localSubHom[0] ** 2, localSubHom[0] * localSubHom[1], localSubHom[1] ** 2, localSubHom))
                    PcoordInt = pCoord + gIntep - np.array([[0], [0], [1]])
                    # 所有点仍然位于目标图像内
                    # if np.prod(PcoordInt[0:2].min(1) > [3, 3]) and np.prod(
                    #         PcoordInt[0:2].min(1) < [self.ref_sizeX - 3, self.ref_sizeY - 3]):
                    # 双三次B样条插值
                    tarIntp = bicubic_Bspline_interp(self.tar_img, PcoordInt)
                    deltagVec = tarIntp - np.mean(tarIntp)
                    deltag = np.sqrt(np.sum(deltagVec ** 2))
                    delta = deltafVec - deltaf / deltag * deltagVec
                    deltap = -inv_H_J @ delta
                    deltap = M @ deltap
                    # 更新扭曲函数
                    deltawarp = p_to_wrap(deltap)
                    warp = warp @ np.linalg.inv(deltawarp)

                    thre = np.sqrt(deltap[0] ** 2 + deltap[6] ** 2)
                    Iter = Iter + 1
                    # 更新参数向量
                    p1 = np.array(
                        [warp[3, 5], warp[3, 3] - 1, warp[3, 4], 2 * warp[3, 0], warp[3, 1], 2 * warp[3, 2],
                         warp[4, 5], warp[4, 4] - 1, warp[4, 3], 2 * warp[4, 0], warp[4, 1],
                         2 * warp[4, 2]]).reshape(-1, 1)

                    Disp = np.array([warp[3, 5], warp[4, 5]])
                    Cznssd = sum(sum((deltafVec / deltaf - deltagVec / deltag) ** 2))
                    Czncc = 1 - 0.5 * Cznssd

                return p1, Czncc, Iter, Disp

            point_disp = cor_point - ref_point
            ref_point_1 = np.append(ref_point, 1).reshape(3, 1).astype('int32')  # 将这个点转换成三行一列，以便中间操作
            if shape_fuc == "一阶形函数":
                p = np.array([point_disp[0], 0, 0, point_disp[1], 0, 0]).reshape(-1, 1)
                p, czncc, Iter, disp = iter_ICGN1(ref_point_1, p)
                if np.isnan(disp).any():
                    return czncc,cor_point
                return czncc, ref_point + disp
            elif shape_fuc == "二阶形函数":
                p = np.array([point_disp[0], 0, 0, 0, 0, 0, point_disp[1], 0, 0, 0, 0, 0]).reshape(-1, 1)
                p, czncc, Iter, disp = iter_ICGN2(ref_point_1, p)  # czncc就是该点匹配的可靠性（也就是匹配程度）
                if np.isnan(disp).any():
                    return czncc,cor_point
                return czncc, ref_point + disp
            else:
                raise ValueError("无效的形函数")

        if method == "曲面拟合法":
            return quadric_surface_fitting_method()
        elif method == "基于灰度梯度搜索":
            return gray_gradient_search()
        elif method == "Newton-Raphson迭代法":
            return newton_raphson_iter()
        else:
            raise ValueError("Unsupported sub_pixel_search method")

    def cal_full_field_displacement(self, integer_method="逐点搜索", cor_criterion="零均值归一化互相关标准",
                                    sub_method="曲面拟合法", shape_func="一阶形函数",
                                    inter_method="双线性插值", reliability=True) -> None:
        """
        基于整数搜索算法和亚像素搜索算法，从初始点开始进行搜索匹配，最后得到全局位移场
        :param integer_method:整数像素搜索方法
        :param cor_criterion: 相关函数选取准则
        :param sub_method: 亚像素搜索方法
        :param shape_func: 形函数选取
        :param inter_method: 插值算法
        :param reliability: 是否采用可靠性引导
        :return: 计算出各计算点的匹配点坐标，n行两列的数组；xy方向的全局位移场，n行两列的二维数组；更新两者信息
        """

        def no_reliability_match():
            DIC.logger.info('使用传统路径法计算全局位移')
            DIC.logger.info(f'总共{DIC.points_num}个计算点')
            ZNCC = np.zeros((DIC.points_num, 1))
            self.field_disp = np.zeros((DIC.points_num, 2))
            self.cor_points = np.zeros((DIC.points_num, 2))
            # 每个点都以初始位移为0进行遍历迭代优化，没有可靠性引导
            for i in range(DIC.points_num_x):
                for j in range(DIC.points_num_y):
                    point_index = i * DIC.points_num_y + j
                    # 传统基于路径传导的方法无需管初始种子点，直接从左上角开始
                    ref_point = DIC.cal_points[point_index, :]
                    int_cor_point = self.integer_pixel_search(ref_point, integer_method, cor_criterion)
                    czncc, sub_cor_point = self.sub_pixel_search(ref_point, int_cor_point, sub_method,
                                                                 shape_func, cor_criterion, inter_method)
                    ZNCC[point_index, :] = czncc
                    sub_cor_point = np.round(sub_cor_point,3)
                    self.cor_points[point_index, :] = sub_cor_point
                    self.field_disp[point_index, :] = sub_cor_point - ref_point
                    DIC.logger.info(f'第{point_index + 1}个点计算完毕，坐标为{ref_point}，匹配点为{sub_cor_point}')

        def reliability_match():
            # 按可靠性引导的路径进行计算
            DIC.logger.info('基于可靠性引导法计算全局位移')
            DIC.logger.info(f'总共{DIC.points_num}个计算点')
            ZNCC = np.zeros((DIC.points_num, 1))
            self.field_disp = np.zeros((DIC.points_num, 2))
            self.cor_points = np.zeros((DIC.points_num, 2))
            # 空队列
            queue = []
            # 用于寻找高可靠性点的四个邻居
            neighbor = np.array([[-1, 0, 1, 0],
                                 [0, -1, 0, 1]])
            # m用来控制顺序
            m = 1
            n = 1

            while queue or m <= 2:
                if m == 1:
                    self.iter_thre = 1e-10  # 迭代阈值
                    int_cor_point = self.integer_pixel_search(DIC.init_point, integer_method, cor_criterion)
                    czncc, sub_cor_point = self.sub_pixel_search(DIC.init_point, int_cor_point, sub_method,
                                                                 shape_func, cor_criterion, inter_method)
                    ZNCC[DIC.index_init_point, :] = czncc
                    self.cor_points[DIC.index_init_point, :] = sub_cor_point
                    self.field_disp[DIC.index_init_point, :] = sub_cor_point - DIC.init_point

                    # 保存初始点的位移
                    self.tar_ref_init_points = self.field_disp.T
                    m = m + 1
                    u, v = np.unravel_index(DIC.index_init_point, [DIC.points_num_x, DIC.points_num_y], 'F')
                    queue.append((u, v, czncc))
                    self.iter_thre = 1e-3

                for neighbor_index in range(4):
                    ii = neighbor[0, neighbor_index]
                    jj = neighbor[1, neighbor_index]
                    i = u + ii
                    j = v + jj
                    if (i < 0 or j < 0 or i > DIC.points_num_x - 1 or j > DIC.points_num_y - 1
                            or ZNCC[i * DIC.points_num_y + j, 0] != 0):
                        continue
                    else:
                        point_index = i * DIC.points_num_y + j

                        ref_point = DIC.cal_points[point_index, :]
                        int_cor_point = self.integer_pixel_search(ref_point, integer_method, cor_criterion)
                        czncc, sub_cor_point = self.sub_pixel_search(ref_point, int_cor_point, sub_method,
                                                                     shape_func, cor_criterion, inter_method)
                        ZNCC[point_index, :] = czncc
                        self.cor_points[point_index, :] = sub_cor_point
                        self.field_disp[point_index, :] = sub_cor_point - ref_point
                        # (i, j)是本次计算点
                        queue.append((i, j, czncc))
                        m = m + 1

                # 队列queue按照Czncc的大小升序排序，高可靠性的点优先计算邻点
                queue.sort(key=lambda x: x[2])
                u = queue[-1][0]
                v = queue[-1][1]
                del queue[-1]
                DIC.logger.info(f'第{n}个点计算完毕，坐标为{ref_point}，匹配点为{sub_cor_point}')
                n = n + 1

        if reliability:
            reliability_match()
        else:
            no_reliability_match()

    def cal_full_field_strain(self, strain_method='直接差分法', strain_type='工程应变') -> None:
        """
        根据全局位移场信息，指定应变场计算方法和应变类型，更新自身全局应变场信息，x方向、y方向线应变和切应变；n行三列
        :param strain_method:选择直接差分法、位移场去噪差分法，基于最小二乘拟合法
        :param strain_type:计算应变类型，有工程应变、柯西应变、格林应变和对数应变
        :return:更新self信息，无返回
        """
        if strain_method == "位移场去噪差分法":
            self.field_disp = cv2.GaussianBlur(self.field_disp, (5, 5), 0)
        # 重构全局位移场，改成形状为（y方向计算点个数，x方向计算点个数）；元素值表示x或y方向位移
        self.to_grid_displacement()
        gradient = np.array(np.gradient(self.disp_grid, axis=(0, 1)))

        def direct_or_smooth_compute_strain():
            # 初始化应变场矩阵,三维数组shape为（m，n，3），前两个表示坐标索引，后一个表示x、y方向线应变和切应变
            self.field_strain = np.zeros((self.disp_grid.shape[0], self.disp_grid.shape[1], 3))

            for i in range(1, self.disp_grid.shape[0] - 1):
                for j in range(1, self.disp_grid.shape[1] - 1):
                    if strain_type == '工程应变':  # 工程应变计算
                        strain_xx = gradient[0, i, j, 0]
                        strain_yy = gradient[1, i, j, 1]
                        strain_xy = 0.5 * (gradient[0, i, j, 1] + gradient[1, i, j, 0])

                    elif strain_type == '对数应变':  # 对数应变计算
                        # 在这里我们计算的是基于一维的简单情况
                        # 对于多维情况，公式会更加复杂
                        strain_xx = np.log(1 + gradient[0, i, j, 0])
                        strain_yy = np.log(1 + gradient[1, i, j, 1])
                        strain_xy = 0  # 对数应变不直观地考虑剪切部分

                    elif strain_type == 'Cauchy应变':  # 柯西应变计算
                        strain_xx = gradient[0, i, j, 0]
                        strain_yy = gradient[1, i, j, 1]
                        strain_xy = (gradient[0, i, j, 1] + gradient[1, i, j, 0]) / 2

                    elif strain_type == 'Green应变':  # 格林应变计算
                        strain_xx = gradient[0, i, j, 0] + 0.5 * (gradient[0, i, j, 0] ** 2 + gradient[0, i, j, 1] ** 2)
                        strain_yy = gradient[1, i, j, 1] + 0.5 * (gradient[0, i, j, 1] ** 2 + gradient[1, i, j, 1] ** 2)
                        strain_xy = (gradient[0, i, j, 1] + gradient[1, i, j, 0]) / 2 + gradient[0, i, j, 0] * gradient[
                            1, i, j, 1]
                    else:
                        raise ValueError("应变类型选取错误！")

                    # 存放计算后的应变值
                    strain_xx = round(strain_xx,3)
                    strain_yy = round(strain_yy,3)
                    strain_xy = round(strain_xy,3)
                    self.field_strain[i, j, 0] = strain_xy
                    self.field_strain[i, j, 1] = strain_yy
                    self.field_strain[i, j, 2] = strain_xy
                    DIC.logger.info(f'计算完{i}行{j}列计算点，应变为{strain_xx, strain_yy, strain_xy}')

        def least_squares_compute_strain_(order=3):
            DIC.logger.info('基于最小二乘拟合法求解全局应变场。')
            # 初始化多项式矩阵
            rows, cols = self.disp_grid[:, :, 0].shape[:2]
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))
            x = x.flatten()
            y = y.flatten()
            A = np.zeros((x.size, (order + 1) ** 2))

            # 构建设计矩阵用于拟合多项式
            idx = 0
            for i in range(order + 1):
                for j in range(order + 1):
                    A[:, idx] = (x ** i) * (y ** j)
                    idx += 1

            # 拟合多项式并计算斜率作为梯度
            gradient_field = np.zeros(self.disp_grid.shape)
            for c in range(self.disp_grid.shape[-1]):
                displacement_flat = self.disp_grid[:, :, c].flatten()
                coeffs, _, _, _ = lstsq(A, displacement_flat)

                # 对于每个点，计算导数
                for i_ in range(order + 1):
                    for j_ in range(order + 1):
                        if order >= i_ + j_ > 0:
                            gradient_field[:, :, c] += coeffs[i_ * (order + 1) + j_] * (
                                    i_ * (x.reshape(rows, cols).astype(float) ** (i_ - 1)) *
                                    (y.reshape(rows, cols).astype(float) ** j_) +
                                    j_ * (x.reshape(rows, cols).astype(float) ** i_) *
                                    (y.reshape(rows, cols).astype(float) ** (j_ - 1)))

            # 初始化应变场矩阵,三维数组shape为（m，n，3），前两个表示坐标索引，后一个表示x、y方向线应变和切应变
            self.field_strain = np.zeros((self.disp_grid.shape[0], self.disp_grid.shape[1], 3))

            for i in range(1, self.disp_grid.shape[0] - 1):
                for j in range(1, self.disp_grid.shape[1] - 1):
                    if strain_type == '工程应变':  # 工程应变计算
                        strain_xx = gradient[0, i, j, 0]
                        strain_yy = gradient[1, i, j, 1]
                        strain_xy = 0.5 * (gradient[0, i, j, 1] + gradient[1, i, j, 0])

                    elif strain_type == '对数应变':  # 对数应变计算
                        # 在这里我们计算的是基于一维的简单情况
                        # 对于多维情况，公式会更加复杂
                        strain_xx = np.log(1 + gradient[0, i, j, 0])
                        strain_yy = np.log(1 + gradient[1, i, j, 1])
                        strain_xy = 0  # 对数应变不直观地考虑剪切部分

                    elif strain_type == 'Cauchy应变':  # 柯西应变计算
                        strain_xx = gradient[0, i, j, 0]
                        strain_yy = gradient[1, i, j, 1]
                        strain_xy = (gradient[0, i, j, 1] + gradient[1, i, j, 0]) / 2

                    elif strain_type == 'Green应变':  # 格林应变计算
                        strain_xx = gradient[0, i, j, 0] + 0.5 * (
                                gradient[0, i, j, 0] ** 2 + gradient[0, i, j, 1] ** 2)
                        strain_yy = gradient[1, i, j, 1] + 0.5 * (
                                gradient[0, i, j, 1] ** 2 + gradient[1, i, j, 1] ** 2)
                        strain_xy = (gradient[0, i, j, 1] + gradient[1, i, j, 0]) / 2 + gradient[0, i, j, 0] * \
                                    gradient[
                                        1, i, j, 1]
                    else:
                        raise ValueError("应变类型选取错误！")

                    # 存放计算后的应变值
                    strain_xx = round(strain_xx, 3)
                    strain_yy = round(strain_yy, 3)
                    strain_xy = round(strain_xy, 3)
                    self.field_strain[i, j, 0] = strain_xy
                    self.field_strain[i, j, 1] = strain_yy
                    self.field_strain[i, j, 2] = strain_xy
                    DIC.logger.info(f'计算完{i}行{j}列计算点，应变为{strain_xx, strain_yy, strain_xy}')

        if strain_method == "位移场去噪差分法" or strain_method == "直接差分法":
            direct_or_smooth_compute_strain()
        elif strain_method == "基于最小二乘拟合法":
            least_squares_compute_strain_()
        else:
            raise ValueError("应变计算方法选取错误！")

    def save_result_to_csv(self, result_directory: str = '.', filename='result.csv') -> None:
        """
        #图像后处理，保存各计算点及对应的匹配点坐标，相对位移量，应变量存储在Excel等格式中
        :param result_directory: 指定目录
        :param filename:保存文件名
        :return: 无
        """
        # 确保传入的参数形状匹配
        a, b, c = self.cor_points, self.field_disp, self.field_strain
        assert DIC.cal_points.shape == self.cor_points.shape == self.field_disp.shape
        m, n, _ = self.field_strain.shape
        assert DIC.cal_points.shape[0] == m * n

        filename = f'{result_directory}' + '/' + f'{filename}'
        if not os.path.exists(result_directory):
            # 如果不存在则创建目录
            os.makedirs(result_directory)

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                '行索引', '列索引', '计算点x坐标', '计算点y坐标', '匹配点x坐标', '匹配点y坐标', 'x位移', 'y位移',
                'strain_xx', 'strain_yy', 'strain_xy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 写入表头
            writer.writeheader()

            # 写入数据
            for i in range(m):
                for j in range(n):
                    # 计算点的一维数组索引
                    index = i * n + j
                    # 将数据写入CSV文件
                    writer.writerow({
                        '行索引': i,
                        '列索引': j,
                        '计算点x坐标': DIC.cal_points[index, 0],
                        '计算点y坐标': DIC.cal_points[index, 1],
                        '匹配点x坐标': self.cor_points[index, 0],
                        '匹配点y坐标': self.cor_points[index, 1],
                        'x位移': self.field_disp[index, 0],
                        'y位移': self.field_disp[index, 1],
                        'strain_xx': self.field_strain[i, j, 0],
                        'strain_yy': self.field_strain[i, j, 1],
                        'strain_xy': self.field_strain[i, j, 2]
                    })


# 完整的算法流程，通过输入的系列图片列表，将前n-1张图片的位移、应变信息都计算出来，保存在DIC对象中，并且保存在指定目录下
def pre_process(img_or_vedio=True, img_path='./img', pre_process_method='ALL') -> List[np.ndarray]:
    """
    图像前处理操作，输入图像数据目录，返回处理好的图像列表
    :param img_or_vedio: 真表示输入为一系列图像；假表示输入为视频
    :param img_path:
    :param pre_process_method:
    :return:返回处理好的图像列表，其中的图像都被cv2读取过
    """
    if img_or_vedio:
        img_list = read_images_from_folder(img_path)
    else:
        img_list = read_video_from_file(img_path, 10)
    assert len(img_list) > 1, "there is not image in " + str(img_path)
    for index, img in enumerate(img_list):
        if pre_process_method == "高斯滤波处理":
            img_list[index] = gaussian_blur(img)
        elif pre_process_method == "灰度消差处理":
            img_list[index] = sobel_filter(img)
        elif pre_process_method == "ALL":
            img_list[index] = sobel_filter(gaussian_blur(img))

        elif pre_process_method == '不需要':
            pass
        else:
            raise ValueError("图像前处理算法选取错误!")

    return img_list


def correlate_parameter_compute(img_list: List[np.ndarray], top_left_p: np.ndarray, bottom_right_p, subset_size=31,
                                step=5,
                                inter_algorithm="双线性插值", cor_criterion="零均值归一化互相关标准",
                                int_search_method="逐点搜索", shape_fuc="一阶形函数", sub_search_method="Newton-Raphson迭代法",
                                reliability=True, log_queue=None) -> List[DIC]:
    """
    图像相关匹配处理，根据设置好的相关参数，对输入进来的图像列表进行相关匹配，返回DIC对象列表，
    其中存储着计算好的全局位移场和全局应变场等信息，并视参数进行保存到指定目录下。
    :param img_list:
    :param top_left_p:
    :param bottom_right_p:
    :param subset_size:
    :param step:
    :param inter_algorithm:
    :param cor_criterion:
    :param int_search_method:
    :param shape_fuc:
    :param sub_search_method:
    :param reliability:
    :param strain_type:
    :param cal_strain_method:
    :param is_saved:
    :param result_path:
    :return:
    """
    DIC.get_calculate_points(top_left_p, bottom_right_p, subset_size, step, log_queue=log_queue)  # 初始化类属性
    ref_img = img_list[0]  # 取第一个图像始终为参考图像
    img_dic_list = []
    i = 1
    for img in img_list[1:]:
        img_dic = DIC(ref_img, img)
        DIC.logger.info(f'第{i}次DIC匹配，对参考图像1和待匹配图像{i + 1}进行')

        img_dic.cal_full_field_displacement(int_search_method, cor_criterion, sub_search_method, shape_fuc,
                                            inter_algorithm, reliability)
        img_dic_list.append(img_dic)
        DIC.logger.info(f'第{i}次匹配成功')

    return img_dic_list


def post_strain_compute(img_list: List[np.ndarray], top_left_p: np.ndarray, bottom_right_p,
                        cor_points_list: List[np.ndarray],
                        displacement_list: List[np.ndarray], subset_size=31, step=5, strain_type='工程应变',
                        cal_strain_method='基于最小二乘拟合法', is_saved=True, result_path='.', log_queue=None):
    i = 1
    img_dic_list = []
    DIC.get_calculate_points(top_left_p, bottom_right_p, subset_size, step, log_queue=log_queue)  # 初始化类属性
    ref_img = img_list[0]  # 取第一个图像始终为参考图像
    for img in img_list[1:]:
        DIC.logger.info(f'开始求解第{i}张图片的全局应变')
        img_dic = DIC(ref_img, img)
        img_dic.cor_points = cor_points_list[i - 1]
        img_dic.field_disp = displacement_list[i - 1]
        img_dic.cal_full_field_strain(cal_strain_method, strain_type)
        DIC.logger.info(f'第{i}张图片的全局应变求解完成')
        if is_saved:
            filename = f'result{i}.csv'
            img_dic.save_result_to_csv(result_path, filename)
            DIC.logger.info(f'result{i}.csv保存成功')
        i += 1
        img_dic_list.append(img_dic)

    return img_dic_list


def post_process_only_result(imgs_path: str, result_path: str):
    """
    基于已经计算好了全场位移和应变，只需要绘制相应图像，单独使用此函数即可
    :param imgs_path:
    :param result_path:
    :return:
    """
    img_list = pre_process(img_path=imgs_path, pre_process_method='不需要')
    pass


if __name__ == "__main__":
    img_path = "D:/study focus/2024本科毕设/dic_yzl/img"
    img_list = pre_process(img_path=img_path)
    h, w = img_list[0].shape[:2]
    start_time = time.time()
    img_dic_list = correlate_parameter_compute(img_list, top_left_p=np.array((0, 0)),
                                               cor_criterion='SQDIFF_NORMED准则',
                                               bottom_right_p=np.array((w // 2, h // 2)), step=5,
                                               shape_fuc='一阶形函数',
                                               int_search_method='逐点搜索', sub_search_method='曲面拟合法',
                                               reliability=False, )
    post_strain_compute(img_list,top_left_p=np.array((0, 0)),bottom_right_p=np.array((w // 2, h // 2)),
                        cor_points_list=[img_dic.cor_points for img_dic in img_dic_list],displacement_list=[img_dic.field_disp for img_dic in img_dic_list])
    end_time = time.time()
    print(end_time - start_time)
    cor_points_list = [img_dic.cor_points for img_dic in img_dic_list]
    displacement_list = [img_dic.field_disp for img_dic in img_dic_list]

    # 存在问题：
    # 1.按照逐点搜索中的零均值归一化方法，对其他方法的点都进行边界内判断,解决
    # 2. 曲面拟合法很不准，后续得看看匹配算法细节
    # 检验：相关准则计算有无问题，解决；整数像素搜素能不能得到一个比较合适的匹配点,解决（后续尝试添加基于粒子群优化算法和遗传算法）
    # ；亚像素搜索精度更高（曲面拟合法解决，但精度略低，基于灰度梯度搜索法解决，牛顿迭代法解决）；全局位移场计算
