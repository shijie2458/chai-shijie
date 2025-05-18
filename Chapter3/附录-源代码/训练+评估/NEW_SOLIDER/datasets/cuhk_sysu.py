import os.path as osp

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset


class CUHKSYSU(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "CUHK-SYSU"
        self.img_prefix = osp.join(root, "Image", "SSM")  # 路径拼接
        super(CUHKSYSU, self).__init__(root, transforms, split)  # 初始化基础数据集，以便在子类中能够正确地继承基础数据集的功能。
    #查询图像信息
    def _load_queries(self):
        # TestG50: a test protocol, 50 gallery images per query
        protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
        protoc = protoc["TestG50"].squeeze()
        queries = []
        for item in protoc["Query"]:
            img_name = str(item["imname"][0, 0][0])
            roi = item["idlocate"][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([-100]),  # dummy pid标签
                }
            )
        return queries
    #数据划分gallery和train
    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        # gallery images
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a[0]) for a in gallery_imgs]
        if self.split == "gallery":
            return gallery_imgs
        # train则是all images-gallery images
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training images = all images - gallery images
        training_imgs = sorted(list(set(all_imgs) - set(gallery_imgs)))
        return training_imgs

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        # load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}  # 创建了两个空字典
        name_to_pids = {}
        unlabeled_pid = 5555  # default pid for unlabeled people
        for img_name, _, boxes in all_imgs:  # _ 在这个上下文中表示一个占位符,不需要使用标签信息
            img_name = str(img_name[0])     # 将图像名称转换为字符串格式
            boxes = np.asarray([b[0] for b in boxes[0]])       # 将boxes列表中的边界框信息转换为 NumPy 数组
            boxes = boxes.reshape(boxes.shape[0], 4)  # 将边界框的维度重塑 (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]  # 找出有效的边界框，其中宽度和高度都大于零
            assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)     #如果有效的边界框索引为空，就抛出断言错误
            boxes = boxes[valid_index]  # 根据有效的边界框索引筛选出有效的边界框
            name_to_boxes[img_name] = boxes.astype(np.int32)  # 将图像名称作为键，将筛选后的有效边界框信息作为值
            name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)  # np.ones用于初始化数组，生成全为 1 的初始值

        def set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):  # 循环遍历一组边界框数组（boxes）中的每个边界框
                if np.all(boxes[i] == box):  #  检查当前边界框是否与给定的 box 完全相等
                    pids[i] = pid  # 是相等的边界框，就将对应的标识符 pid 设置到对应的标识符数组 pids
                    return

        # assign a unique pid from 1 to N for each identity
        # 训练集
        if self.split == "train":
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for img_name, box, _ in scenes:
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)
        # 测试
        else:
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)

        annotations = []
        imgs = self._load_split_img_names()
        for img_name in imgs:
            boxes = name_to_boxes[img_name]
            boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)边界框表示方式转换+宽高
            pids = name_to_pids[img_name]
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": boxes,
                    "pids": pids,
                }
            )
        return annotations
