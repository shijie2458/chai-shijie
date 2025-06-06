from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils

from models.oim_arcface import OIMLoss   # 改损失函数
# from models.oim import OIMLoss
from models.resnet import build_resnet   # backbone二选一
from models.swin import build_swin

class SeqNet(nn.Module):
    def __init__(self, cfg):
        super(SeqNet, self).__init__()

        backbone_name = cfg.MODEL.BONE                                            # swin_tiny
        semantic_weight = cfg.MODEL.SEMANTIC_WEIGHT                               # 语义权重0.6
        if backbone_name == 'resnet50':
            backbone, box_head = build_resnet(name="resnet50", pretrained=True)   # Res5Head
            feat_len = 2048
        elif 'swin' in backbone_name:
            backbone, box_head, feat_len = build_swin(name=backbone_name, semantic_weight=semantic_weight)  # （384，384，768）

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        faster_rcnn_predictor = FastRCNNPredictor(feat_len, 2)                   # 输入特征维度2048、numclass=2前景背景
        reid_head_2nd = deepcopy(box_head)                                      # 创建完全独立于 box_head 的新对象 reid_head
        reid_head_3rd = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2   # 在resnet4后面加ROI，输出特征矩阵14*14
        )
        box_predictor = BBoxRegressor(feat_len, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        roi_heads = SeqRoIHeads(
            cfg=cfg,
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,           # 5532
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,         # 5000
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,   # 0.5
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,       # 30
            arcface_loss_weight= cfg.MODEL.LOSS.ARCFACE_LOSS_WEIGHT,  #修改参数
            cosine=cfg.MODEL.LOSS.COSINE,             #修改参数
            # SeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head_2nd=reid_head_2nd,
            reid_head_3rd=reid_head_3rd,
            # parent class                                              # Fastrcnn原本的传入参数
            box_roi_pool=box_roi_pool,                                  # ROI Align定位更准
            box_head=box_head,                                          # Res5Head
            box_predictor=box_predictor,                                # 两个全连接层分类回归
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,          # 0.5前景背景正负样本
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,   # 每张图片选取128个proposal，RPN提供2000个
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,        # 正样本所占比例0.5
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
            feat_len=feat_len,
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        self.eval_feat = cfg.EVAL_FEATURE

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID
        self.lw_rcnn_reg_3rd = cfg.SOLVER.LW_RCNN_REG_3RD
        self.lw_rcnn_cls_3rd = cfg.SOLVER.LW_RCNN_CLS_3RD
        self.lw_rcnn_reid_3rd = cfg.SOLVER.LW_RCNN_REID_3RD



    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]           # 表示获取图像的高度和宽度
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)                            # 模型需要图像表示为张量计算

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features_2nd = self.roi_heads.reid_head_2nd(box_features)
            embeddings_2nd, _ = self.roi_heads.embedding_head_2nd(box_features_2nd)         # 特征归一化NAE
            box_features_3rd = self.roi_heads.reid_head_3rd(box_features)
            embeddings_3rd, _ = self.roi_heads.embedding_head_3rd(box_features_3rd)
            # 选择不同的特征进行评估
            if self.eval_feat == 'concat':
                embeddings = torch.cat((embeddings_2nd, embeddings_3rd), dim=1)
            elif self.eval_feat == 'stage2':
                embeddings = embeddings_2nd
            elif self.eval_feat == 'stage3':
                embeddings = embeddings_3rd
            else:
                raise Exception("Unknown evaluation feature name")
            return embeddings.split(1, 0)                                       # 将张量分割成多个小块
        else:
            # gallery
            proposals, _ = self.rpn(images, features, targets)
            # detections, _ = self.roi_heads(
            #     features, proposals, images.image_sizes, targets, query_img_as_gallery
            # )
            detections = self.roi_heads(
                features, proposals, images.image_sizes, targets, query_img_as_gallery
            )[0]
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)

        images, targets = self.transform(images, targets)       # images类型ImageList
        features = self.backbone(images.tensors)            # images[2,3,1216,1504]>>[B,C,H,W]每张图片长宽不一样
        # backbone返回字典类型 ，打印OrderedDict对象的键值对
        proposals, proposal_losses = self.rpn(images, features, targets)

        _, detector_losses, feats_reid_2nd, targets_reid_2nd, feats_reid_3rd, targets_reid_3rd = self.roi_heads(features, proposals, images.image_sizes, targets)  # SeqRoIHeads

        # rename rpn losses to be consistent with detection losses，                     # 将proposal_losses字典中的键名进行更改
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}  # 将两字典中的键值对合并到一个新的字典
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        losses["loss_rcnn_reid_2nd"] *= self.lw_box_reid
        losses["loss_rcnn_reg_3rd"] *= self.lw_rcnn_reg_3rd
        losses["loss_rcnn_cls_3rd"] *= self.lw_rcnn_cls_3rd
        losses["loss_rcnn_reid_3rd"] *= self.lw_rcnn_reid_3rd

        return losses, feats_reid_2nd, targets_reid_2nd, feats_reid_3rd, targets_reid_3rd


class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        cfg,
        num_pids,               # 有标签的
        num_cq_size,            # 没有标签的
        oim_momentum,
        oim_scalar,
        arcface_loss_weight,
        cosine,
        faster_rcnn_predictor,
        reid_head_2nd,
        reid_head_3rd,
        feat_len,
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)

        # ROI head使用不同阈值
        self.use_diff_thresh= cfg.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_1ST
        self.nms_thresh_2nd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.nms_thresh_3rd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_3RD
        self.fg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND
        self.fg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_3RD
        self.bg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_3RD

        self.embedding_head_2nd = NormAwareEmbedding(in_channels=[int(feat_len/2),feat_len], dim=256)
        self.embedding_head_3rd = deepcopy(self.embedding_head_2nd)

        self.reid_loss_2nd = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar, arcface_loss_weight, cosine) # 全部arcface
        # self.reid_loss_2nd = OIMLoss(256, num_pids, num_cq_size, oim_momentum, oim_scalar)
        self.reid_loss_3rd = deepcopy(self.reid_loss_2nd)

        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.box_predictor_2nd = self.box_predictor
        self.box_predictor_3rd = deepcopy(self.box_predictor)

        self.reid_head_2nd = reid_head_2nd
        self.reid_head_3rd = reid_head_3rd
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections
        # evaluation
        self.eval_feat = cfg.EVAL_FEATURE

    def forward(self, features, proposals, image_shapes, targets=None, query_img_as_gallery=False):
        """
        Arguments:
            features (List[Tensor])图像特征
            proposals (List[Tensor[N, 4]])RPN得到的提议框
            image_shapes (List[Tuple[H, W]])预处理得到的图片大小，等比例缩放
            targets (List[Dict])真实目标标注信息
        """
        cws = True
        gt_det_2nd = None
        gt_det_3rd = None
        feats_reid_2nd = None
        feats_reid_3rd = None
        targets_reid_2nd = None
        targets_reid_3rd = None

        if self.training:   # 划分正负样本，统计对应GT的标签以及边界框回归信息（单图RPN2000个采样128个）
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False)
            proposals, _, proposal_pid_labels, proposal_reg_targets = self.select_training_samples(
                proposals, targets
            )

        # ------------------- Faster R-CNN head ------------------ #
        proposal_features = self.box_roi_pool(features, proposals, image_shapes)    # ROI align
        proposal_features = self.box_head(proposal_features)                        # res5
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res5"]                                          # 2048,访问字典"feat_res5",分类回归
        )

        if self.training:
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False)
            # detach()新的张量,不参与梯度运算
            boxes, _, box_pid_labels_2nd, box_reg_targets_2nd = self.select_training_samples(boxes, targets)        # <class 'list'>
            # print("box_pid_labels_2nd",box_pid_labels_2nd)
            # print(type(box_pid_labels_2nd))
        else:
            # invoke the postprocess method inherited from parent class to process proposals
            orig_thresh = self.nms_thresh  # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )

        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.reid_head_2nd(gt_box_features)
            embeddings, _ = self.embedding_head_2nd(gt_box_features)
            gt_det_2nd = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:      # 测试阶段没有检测结果可用
            assert not self.training
            boxes = gt_det_2nd["boxes"] if gt_det_2nd else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            if self.eval_feat == 'concat':      # 将两个特征拼接512维度
                embeddings = torch.cat((gt_det_2nd["embeddings"], gt_det_2nd["embeddings"]), dim=1) if gt_det_2nd else torch.zeros(0, 512)
            elif self.eval_feat == 'stage2' or self.eval_feat == 'stage3':
                embeddings = gt_det_2nd["embeddings"] if gt_det_2nd else torch.zeros(0, 256)
            else:
                raise Exception("Unknown evaluation feature name")
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- 2Baseline head -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head_2nd(box_features)                             # reid=res5
        box_regs_2nd = self.box_predictor_2nd(box_features["feat_res5"])                # 只生成框2048d
        box_embeddings_2nd, box_cls_scores_2nd = self.embedding_head_2nd(box_features)      # 分类和Reid
        if box_cls_scores_2nd.dim() == 0:                                           # 如果 box_cls_scores 的维度为 0（标量），则添加一个维度
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        if self.training:
            boxes = self.get_boxes(box_regs_2nd, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_3rd,
                    self.bg_iou_thresh_3rd,
                    allow_low_quality_matches=False)
            boxes, _, box_pid_labels_3rd, box_reg_targets_3rd = self.select_training_samples(boxes, targets)
        else:
            self.nms_thresh = self.nms_thresh_2nd
            if self.eval_feat != 'stage2':
                boxes, scores, _, _ = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=None,
                    cws=cws,
                )

        if not self.training and query_img_as_gallery and self.eval_feat != 'stage2':
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.reid_head_3rd(gt_box_features)
            embeddings, _ = self.embedding_head_3rd(gt_box_features)
            gt_det_3rd = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0 and self.eval_feat != 'stage2':      # 测试阶段没有检测结果可用
            assert not self.training
            boxes = gt_det_3rd["boxes"] if gt_det_3rd else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
            if self.eval_feat == 'concat':
                embeddings = torch.cat((gt_det_2nd["embeddings"], gt_det_3rd["embeddings"]), dim=1) if gt_det_3rd else torch.zeros(0, 512)
            elif self.eval_feat == 'stage3':
                embeddings = gt_det_2nd["embeddings"] if gt_det_3rd else torch.zeros(0, 256)
            else:
                raise Exception("Unknown evaluation feature name")
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- The third stage  -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)

        if not self.training:
            box_features_2nd = self.reid_head_2nd(box_features)
            box_embeddings_2nd, _ = self.embedding_head_2nd(box_features_2nd)

        box_features = self.reid_head_3rd(box_features)                                     # reid=res5
        box_regs_3rd = self.box_predictor_3rd(box_features["feat_res5"])                    # 只生成框2048d
        box_embeddings_3rd, box_cls_scores_3rd = self.embedding_head_3rd(box_features)      # 分类和Reid
        if box_cls_scores_3rd.dim() == 0:                                                   # 如果 box_cls_scores 的维度为 0（标量），则添加一个维度
            box_cls_scores_3rd = box_cls_scores_3rd.unsqueeze(0)

        result, losses = [], {}
        if self.training:                                                                   # 训练阶段计算损失
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            box_labels_3rd = [y.clamp(0, 1) for y in box_pid_labels_3rd]
            losses = detection_losses(
                # 第一阶段
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
                # 第二阶段
                box_cls_scores_2nd,
                box_regs_2nd,
                box_labels_2nd,
                box_reg_targets_2nd,
                # 第三阶段
                box_cls_scores_3rd,
                box_regs_3rd,
                box_labels_3rd,
                box_reg_targets_3rd,
            )
            loss_rcnn_reid_2nd, feats_reid_2nd, targets_reid_2nd = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd)
            loss_rcnn_reid_3rd, feats_reid_3rd, targets_reid_3rd = self.reid_loss_3rd(box_embeddings_3rd, box_pid_labels_3rd)
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            losses.update(loss_rcnn_reid_3rd=loss_rcnn_reid_3rd)
        else:
            # orig_thresh = self.nms_thresh
            # self.nms_thresh = 0.5                                           # 测试阶段,非极大值抑制（NMS）的阈值
            if self.eval_feat == 'stage2':
                boxes, scores, embeddings_2nd, labels = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=gt_det_2nd,
                    cws=cws,
                )
            else:
                self.nms_thresh = self.nms_thresh_3rd
                _, _, embeddings_2nd, _ = self.postprocess_boxes(
                    box_cls_scores_3rd,
                    box_regs_3rd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=gt_det_2nd,
                    cws=cws,
                )
                boxes, scores, embeddings_3rd, labels = self.postprocess_boxes(
                    box_cls_scores_3rd,
                    box_regs_3rd,
                    box_embeddings_3rd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=gt_det_3rd,
                    cws=cws,
                )
                # set to original thresh after finishing postprocess
                self.nms_thresh = orig_thresh

            num_images = len(boxes)
            for i in range(num_images):
                if self.eval_feat == 'concat':
                    embeddings = torch.cat((embeddings_2nd[i], embeddings_3rd[i]), dim=1)
                elif self.eval_feat == 'stage2':
                    embeddings = embeddings_2nd[i]
                elif self.eval_feat == 'stage3':
                    embeddings = embeddings_3rd[i]
                else:
                    raise Exception("Unknown evaluation feature name")
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings
                    )
                )
        return result, losses, feats_reid_2nd, targets_reid_2nd, feats_reid_3rd, targets_reid_3rd


    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.最终的目标边界框
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]     # 计算每个图像中的候选框数量，将其存储在列表
        pred_boxes = self.box_coder.decode(box_regression, proposals)               # 预测的边界框
        pred_boxes = pred_boxes.split(boxes_per_image, 0)                           # 将预测边界框 pred_boxes 按照每个图像的候选框数量进行切分，得到一个列表

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)                 # 将边界框裁剪到图像边界内
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)                                     # 将剩余的边界框重新形状为二维张量
            all_boxes.append(boxes)

        return all_boxes
    # 处理目标检测模型的输出，执行非极大值抑制（NMS），并筛选出最终的检测结果
    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)       # 将预测的边界框位置（proposal）解码为真实图像上的坐标

        if fcs is not None:
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head_2nd.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels

# NAE
class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):    # 输入一个为1024，2048，输出通道256
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()                                                           # 将输入特征映射投影到低维空间中
        indv_dims = self._split_embedding_dim()                                                     # indv_dims 存储了每个输入特征映射的嵌入维度
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
    # 将输入张量 x 进行扁平化处理
    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x
    # 将给定的嵌入维度（self.dim）分割成若干部分
    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp

# 边界框回归
class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:                                              # 线性层和批归一化层
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)    # 初始化线性层和批归一化层的权重
            init.normal_(self.bbox_pred[1].weight, std=0.01)    # 初始化线性层和批归一化层的偏置
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)    # 不使用批归一化，创建一个只包含线性层的模型
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)     # 自适应平均池化
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas

# 目标检测中的损失函数计算
def detection_losses(
    proposal_cls_scores,            # 候选框
    proposal_regs,
    proposal_labels,
    proposal_reg_targets,
    box_cls_scores_2nd,                 # 边界框
    box_regs_2nd,
    box_labels_2nd,
    box_reg_targets_2nd,
    box_cls_scores_3rd,             # 第三阶段
    box_regs_3rd,
    box_labels_3rd,
    box_reg_targets_3rd,

):
    # --------------------- The first stage -------------------- #
    proposal_labels = torch.cat(proposal_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)
    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)               # 交叉熵损失函数

    # 筛选出正样本（有物体的样本）的索引，然后计算回归损失
    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(                                   # 平滑的 L1 损失函数
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    # --------------------- The second stage -------------------- #
    box_labels_2nd = torch.cat(box_labels_2nd, dim=0)
    box_reg_targets_2nd = torch.cat(box_reg_targets_2nd, dim=0)
    # print("box_labels_2nd",box_labels_2nd.shape)  # box_labels_2nd: torch.Size([256])
    # print("box_cls_scores_2nd",box_cls_scores_2nd.shape)  #  box_cls_scores_2nd:torch.Size([256])
    loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores_2nd, box_labels_2nd.float())   # 二元交叉熵损失

    sampled_pos_inds_subset = torch.nonzero(box_labels_2nd > 0).squeeze(1)
    labels_pos = box_labels_2nd[sampled_pos_inds_subset]
    N = box_cls_scores_2nd.size(0)
    box_regs_2nd = box_regs_2nd.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs_2nd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_2nd[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels_2nd.numel()

    # --------------------- The third stage -------------------- #
    box_labels_3rd = torch.cat(box_labels_3rd, dim=0)
    box_reg_targets_3rd = torch.cat(box_reg_targets_3rd, dim=0)
    loss_rcnn_cls_3rd = F.binary_cross_entropy_with_logits(box_cls_scores_3rd, box_labels_3rd.float())

    sampled_pos_inds_subset = torch.nonzero(box_labels_3rd > 0).squeeze(1)
    labels_pos = box_labels_3rd[sampled_pos_inds_subset]
    N = box_cls_scores_3rd.size(0)
    box_regs_3rd = box_regs_3rd.reshape(N, -1, 4)

    loss_rcnn_reg_3rd = F.smooth_l1_loss(
        box_regs_3rd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_3rd[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_rcnn_reg_3rd = loss_rcnn_reg_3rd / box_labels_3rd.numel()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
        loss_rcnn_cls_3rd=loss_rcnn_cls_3rd,
        loss_rcnn_reg_3rd=loss_rcnn_reg_3rd,
    )
