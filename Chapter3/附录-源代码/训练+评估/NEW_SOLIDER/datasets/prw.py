import os.path as osp
import re

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset


class PRW(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "PRW"
        self.img_prefix = osp.join(root, "frames")
        super(PRW, self).__init__(root, transforms, split)

    def _get_cam_id(self, img_name):  # 从图像名称中提取相机标识符
        match = re.search(r"c\d", img_name).group().replace("c", "")  # 使用正则表达式（re.search）搜索图像名称中的形如 "c\d" 的相机标识符部分 从匹配结果中去除 "c"，得到相机标识符
        return int(match)

    def _load_queries(self):
        query_info = osp.join(self.root, "query_info.txt")
        with open(query_info, "rb") as f:  # 将文件query_info 打开，使用二进制模式（"rb"）读取其内容
            raw = f.readlines()

        queries = []
        for line in raw:
            linelist = str(line, "utf-8").split(" ")  # 字节串是以二进制形式存储的文本UTF-8 编码转换为字符串，split(" ")：使用空格作为分隔符
            pid = int(linelist[0])
            x, y, w, h = (
                float(linelist[1]),
                float(linelist[2]),
                float(linelist[3]),
                float(linelist[4]),
            )
            roi = np.array([x, y, x + w, y + h]).astype(np.int32)  # np.array：创建一个 NumPy 数组
            roi = np.clip(roi, 0, None)  # several coordinates are negative，np.clip：用于将数组中的元素限制在指定的范围内。下限不能小于0，无上限。
            img_name = linelist[5][:-2] + ".jpg"   # 获取第五个元素，切片（slice）来截取字符串
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([pid]),
                    "cam_id": self._get_cam_id(img_name),
                }
            )
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        if self.split == "train":
            imgs = loadmat(osp.join(self.root, "frame_train.mat"))["img_index_train"]
        else:
            imgs = loadmat(osp.join(self.root, "frame_test.mat"))["img_index_test"]
        return [img[0][0] + ".jpg" for img in imgs]

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        annotations = []
        imgs = self._load_split_img_names()
        for img_name in imgs:
            anno_path = osp.join(self.root, "annotations", img_name)
            anno = loadmat(anno_path)
            box_key = "box_new"  # 设置默认的关键字，用于获取边界框信息
            if box_key not in anno.keys():
                box_key = "anno_file"
            if box_key not in anno.keys():
                box_key = "anno_previous"

            rois = anno[box_key][:, 1:]  # 从注释文件中获取边界框信息（去除第一列，即人物标识符）
            ids = anno[box_key][:, 0]  # 从注释文件中获取人物标识符信息（第一列）
            rois = np.clip(rois, 0, None)  # several coordinates are negative使用 np.clip 函数将边界框坐标限制在非负范围内，以确保坐标不会小于零

            assert len(rois) == len(ids)

            rois[:, 2:] += rois[:, :2]  # 将每个边界框的宽度（w）和高度（h）与左上角坐标（x1, y1）进行相加
            ids[ids == -2] = 5555  # assign pid = 5555 for unlabeled people将人物标识符中为 -2 的值设置为 5555，用于为未标记的人物分配一个特定的标识符。
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": rois.astype(np.int32),
                    # FIXME: (training pids) 1, 2,..., 478, 480, 481, 482, 483, 932, 5555
                    "pids": ids.astype(np.int32),
                    "cam_id": self._get_cam_id(img_name),
                }
            )
        return annotations
