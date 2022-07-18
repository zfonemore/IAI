import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, kaiming_init
from mmcv.runner import force_fp32

from mmdet.core import (distance2bbox, bbox2result_with_id)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from .utils import multiclass_nms, parse_dynamic_params, compute_locations, aligned_bilinear

INF = 1e8

@HEADS.register_module()
class IAICondInstHead(AnchorFreeHead):
    """IAICondInstHead
       Add new ID Head & new ID loss to original CondInst head
    """
    def __init__(self,
                 num_classes,
                 max_obj_num,
                 in_channels,
                 stacked_convs=4,
                 strides=[8, 16, 32, 64, 128],
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_id=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 **kwargs):
        self.max_obj_num = max_obj_num
        self.id_out_channels = self.max_obj_num
        super(IAICondInstHead, self).__init__(num_classes, in_channels, **kwargs)
        self.strides = strides
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_on_bbox = norm_on_bbox

        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)

        # add new ID loss
        if loss_id is not None:
            self.loss_id = build_loss(loss_id)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head.
           Add new ID Head consists of two 1x1 convolution layers
           ID Head = id_convs (1xconv) + conv_id (1xconv)
        """

        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.id_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            if i == 0:
                self.id_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.conv_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.conv_id = nn.Conv2d(
            self.feat_channels,
            self.id_out_channels,
            3,
            padding=1)
        self.conv_reg = nn.Conv2d(
            self.feat_channels,
            4,
            3,
            padding=1)
        self.conv_centerness = nn.Conv2d(
            self.feat_channels,
            1,
            3,
            padding=1)
        self.controller = nn.Conv2d(
            self.feat_channels,
            169,
            3,
            padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])
        # mask branch
        self.mask_refine = nn.ModuleList()
        in_features = ['p3', 'p4', 'p5']
        for in_feature in in_features:
            conv_block = []
            conv_block.append(
                nn.Conv2d(self.feat_channels,
                          128,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            conv_block.append(nn.BatchNorm2d(128))
            conv_block.append(nn.ReLU())
            conv_block = nn.Sequential(*conv_block)
            self.mask_refine.append(conv_block)
        # mask head
        tower = []
        for i in range(self.stacked_convs):
            conv_block = []
            conv_block.append(
                nn.Conv2d(128,
                          128,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            conv_block.append(nn.BatchNorm2d(128))
            conv_block.append(nn.ReLU())

            conv_block = nn.Sequential(*conv_block)
            tower.append(conv_block)

        tower.append(
            nn.Conv2d(128,
                      8,
                      kernel_size=1,
                      stride=1))
        self.mask_head = nn.Sequential(*tower)

        # conditional convs
        self.weight_nums = [80, 64, 8]
        self.bias_nums = [8, 8, 1]
        self.mask_out_stride = 4

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.id_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_id, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_centerness, std=0.01)
        kaiming_init(self.mask_refine)
        kaiming_init(self.mask_head)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is num_classes.
                id_scores (list[Tensor]): ID scores for all scale
                    levels, each is a 4D-tensor, the channels number is max_obj_num.
                bbox_preds (list[Tensor]): Box energies / deltas for all scale
                    levels, each is a 4D-tensor, the channels number is 4.
        """
        id_scores = []
        cls_scores = []
        bbox_preds = []
        centernesses = []
        kernel_preds = []
        for i, (x, scale) in enumerate(zip(feats, self.scales)):
            cls_feat = x
            id_feat = x
            reg_feat = x

            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for id_conv in self.id_convs:
                id_feat = id_conv(id_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

            cls_score = self.conv_cls(cls_feat)
            id_score = self.conv_id(id_feat)
            bbox_pred = scale(self.conv_reg(reg_feat)).float()
            if self.norm_on_bbox:
                bbox_pred = F.relu(bbox_pred) * self.strides[i]
            else:
                bbox_pred = bbox_pred.exp()
            centerness = self.conv_centerness(reg_feat)
            kernel_pred = self.controller(reg_feat)

            # mask feat
            if i == 0:
                mask_feat = self.mask_refine[i](x)
            elif i <= 2:
                x_p = self.mask_refine[i](x)
                target_h, target_w = mask_feat.size()[2:]
                h, w = x_p.size()[2:]
                assert target_h % h == 0
                assert target_w % w == 0
                factor_h, factor_w = target_h // h, target_w // w
                assert factor_h == factor_w
                x_p = aligned_bilinear(x_p, factor_h)
                mask_feat = mask_feat + x_p

            bbox_preds.append(bbox_pred)
            cls_scores.append(cls_score)
            id_scores.append(id_score)
            centernesses.append(centerness)
            kernel_preds.append(kernel_pred)

        mask_feat = self.mask_head(mask_feat)

        return cls_scores, id_scores, bbox_preds, centernesses, kernel_preds, mask_feat

    @force_fp32(apply_to=('cls_scores', 'id_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             id_scores,
             bbox_preds,
             centernesses,
             kernel_preds,
             mask_feats,
             gt_bboxes,
             gt_labels,
             gt_ids,
             img_metas,
             gt_bboxes_ignore=None,
             is_first=False):
        """Compute losses of the head.
        Not available for evaluation code

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            id_scores (list[Tensor]): ID scores for each scale level
                Has shape (N, max_obj_num, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale
                level with shape (N, 1, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_ids (Tensor) : Ground truth IDs of each box,
                shape (num_gts,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_ids=None,
                      proposal_cfg=None,
                      **kwargs):
        """Training forward of the head
        Not available for evaluation code
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_masks (Tensor) : Ground truth masks of each box.
            gt_ids (Tensor) : Ground truth IDs of each box,
                shape (num_gts,).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            losses: (dict[str, Tensor]): A dictionary of loss components.
        """
        pass

        return None

    def mask_heads_forward(self, features, weights, biases, num_instances):
        '''Mask head forward process'''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x,
                         w,
                         bias=b,
                         stride=1,
                         padding=0,
                         groups=num_instances)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def relative_coordinate_feature_generator(self, mask_feat, instance_locations, strides):
        # obtain relative coordinate features for mask generator
        num_instance = len(instance_locations)
        H, W = mask_feat.size()[1:]
        locations = compute_locations(H,
                                      W,
                                      stride=8,
                                      device=mask_feat.device)
        relative_coordinates = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
        relative_coordinates = relative_coordinates.permute(0, 2, 1).float()
        relative_coordinates = relative_coordinates / (strides.float().reshape(-1, 1, 1) * 8.0)
        relative_coordinates = relative_coordinates.to(dtype=mask_feat.dtype)
        coordinates_feat = torch.cat([
            relative_coordinates.view(num_instance, 2, H, W),
            mask_feat.repeat(num_instance, 1, 1, 1)], dim=1)
        coordinates_feat = coordinates_feat.view(1, -1, H, W)
        return coordinates_feat

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   id_scores,
                   bbox_preds,
                   centernesses,
                   kernel_preds,
                   mask_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   is_first=False):
        """Transform network output for a batch into bbox, mask & ID predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_classes, H, W).
            id_scores (list[Tensor]): ID scores for each scale level
                Has shape (N, max_obj_num, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            is_first (bool): Whether this is the first frame of the video,
                if true, do some specific operations
                Default: False.

        Returns:
            bbox_results[dict]: a dict of bbox results

            segm_results[dict] : a dict of segmentation results

            id_masks[Tensor] : one hot ID masks, is used to create ID embedding

            new_inst_exists[bool] :  whether a new instance exists in current frame, is used to update global memory

            return_cls_scores[dict] : an ensemble dict of classification category & score for each instance in the video
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds)

        # initial ID numbers for a new video
        if is_first:
            self.curr_inst_id = [0 for i in range(len(img_metas))]
        mask_results_list = []

        max_h, max_w = 0, 0
        for img_id in range(len(img_metas)):
            pad_shape = img_metas[img_id]['pad_shape']
            h, w = pad_shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        id_masks = cls_scores[0].new_zeros(len(img_metas), self.max_obj_num+1, max_h, max_w)
        id_masks[:, self.max_obj_num] = 1
        new_inst_exists = False

        # get multiple level points for all featmaps from different scales
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points, mlvl_strides = self.get_points(featmap_sizes, bbox_preds[0].dtype,
            bbox_preds[0].device)

        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            id_score_list = [
                id_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            kernel_pred_list = [
                kernel_preds[i][img_id].detach() for i in range(num_levels)
            ]
            mask_feats_i = mask_feats[img_id]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            # get predict bboxes, ids, classication scores, ori_masks(for output results, original image shape), det_masks(for association, input shape)
            det_bboxes, det_id_scores, det_cls_scores, ori_masks, det_masks = self._get_bboxes_single(
                cls_score_list,
                id_score_list,
                bbox_pred_list,
                centerness_pred_list,
                kernel_pred_list,
                mask_feats_i,
                mlvl_points,
                mlvl_strides,
                img_shape,
                scale_factor,
                ori_shape,
                cfg,
                rescale,
                with_nms,
                is_first=is_first)

            # remove some wrong masks (no positive area or occluded by previous masks)
            keep = []
            new_pad_masks = []
            bg_mask = id_masks.new_ones((max_h, max_w), dtype=torch.bool)
            if ori_masks is not None:
                ori_bg_mask = id_masks.new_ones(ori_masks[0].shape, dtype=torch.bool)
                new_ori_masks = []

            for i in range(len(det_bboxes)):
                mask = det_masks[i].bool()
                pad = (0, max_w - mask.shape[1], 0, max_h - mask.shape[0])
                pad_mask = F.pad(mask, pad, value=0)
                new_pad_mask = bg_mask & pad_mask
                bg_mask[new_pad_mask] = 0
                area = pad_mask.sum()
                if (area == 0) or float(new_pad_mask.sum()) / float(area) < 0.1:
                    continue
                keep.append(i)
                new_pad_masks.append(new_pad_mask)

                if ori_masks is not None:
                    new_ori_mask = ori_bg_mask & ori_masks[i].bool()
                    ori_bg_mask[new_ori_mask] = 0
                    new_ori_masks.append(new_ori_mask)

            if len(keep) > 0:
                det_bboxes = det_bboxes[keep]
                det_id_scores = det_id_scores[keep]
                if ori_masks is not None:
                    ori_masks = ori_masks[keep]
                det_cls_scores = det_cls_scores[keep]

            # hungarian algorithm to assign unique ID for each instance
            from scipy.optimize import linear_sum_assignment
            if is_first:
                det_ids = det_id_scores.new_ones(det_bboxes.shape[0]) * (self.max_obj_num - 1)
            else:
                if len(det_id_scores) > 0:
                    new_id_scores = det_id_scores[:, self.max_obj_num-1].repeat(len(det_id_scores)-1,1)
                    id_scores_matrix = -torch.cat((det_id_scores.transpose(0,1), new_id_scores)).transpose(0,1).cpu()
                    row_ind, col_ind = linear_sum_assignment(id_scores_matrix)
                    det_ids = col_ind

            # combine ID with masks to generate ID masks
            det_obj_ids = []
            cls_scores_dict = {}
            curr_max_id = self.curr_inst_id[img_id]
            for i in range(len(keep)):
                id_pred = det_ids[i].item()
                new_pad_mask = new_pad_masks[i]

                # in first frame or in the following frames new instance exists
                if (is_first) or (id_pred >= curr_max_id):
                    new_inst_exists = True
                    id_pred = self.curr_inst_id[img_id]
                    self.curr_inst_id[img_id] += 1
                    # if instance ID number surpass the maximum object numbers, ignore this object
                    if self.curr_inst_id[img_id] > self.max_obj_num-2:
                        self.curr_inst_id[img_id] = self.max_obj_num-2

                if id_pred in det_obj_ids:
                    id_pred = self.curr_inst_id[img_id]
                    self.curr_inst_id[img_id] += 1
                    if self.curr_inst_id[img_id] > self.max_obj_num-2:
                        self.curr_inst_id[img_id] = self.max_obj_num-2

                det_obj_ids.append(id_pred)
                cls_scores_dict[id_pred] = det_cls_scores[i]
                id_masks[img_id][id_pred] = new_pad_mask
                id_masks[img_id][self.max_obj_num][new_pad_mask] = 0

            if len(keep) == 0:
                bbox_results = {}
                mask_results_list.append({})
                continue

            bbox_results = bbox2result_with_id(det_bboxes, det_bboxes.new_ones(det_bboxes.size(0)), det_obj_ids)

            mask_results = {}
            for i in range(len(keep)):
                id_pred = det_obj_ids[i]
                mask = new_ori_masks[i].bool().cpu().numpy()
                mask_results[id_pred] = mask
            mask_results_list.append(mask_results)

        return bbox_results, mask_results_list, id_masks, new_inst_exists, cls_scores_dict

    def _get_bboxes_single(self,
                           cls_scores,
                           id_scores,
                           bbox_preds,
                           centernesses,
                           kernel_preds,
                           mask_feat,
                           mlvl_points,
                           mlvl_strides,
                           img_shape,
                           scale_factor,
                           ori_shape,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           is_first=False):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single
                scale level with shape (4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (1, H, W).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            det_bboxes (list[Tensor]): a list of predicted bboxes.
            det_id_scores (list[Tensor]) : a list of predicted ID scores.
            det_cls_scores (list[Tensor]) : a list of predicted classification scores.
            ori_masks (list[Tensor]) : a list of predicted masks of original size.
            masks (list[Tensor]) : a list of predicted masks of input size.
        """
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_id_scores = []
        mlvl_centerness = []
        mlvl_kernels_pred = []
        flatten_mlvl_points = []
        flatten_mlvl_strides = []
        for cls_score, id_score, bbox_pred, centerness, kernel_pred, points, strides in zip(
                cls_scores, id_scores, bbox_preds, centernesses, kernel_preds,  mlvl_points, mlvl_strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            id_scores_pred = id_score.permute(1, 2, 0).reshape(
                -1, self.id_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            kernel_pred = kernel_pred.permute(1, 2, 0).reshape(-1, 169)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)

                points = points[topk_inds, :]
                strides = strides[topk_inds]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                id_scores_pred = id_scores_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                kernel_pred = kernel_pred[topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_id_scores.append(id_scores_pred)
            mlvl_centerness.append(centerness)
            mlvl_kernels_pred.append(kernel_pred)
            flatten_mlvl_strides.append(strides)
            flatten_mlvl_points.append(points)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_kernels_pred = torch.cat(mlvl_kernels_pred)

        flatten_mlvl_points = torch.cat(flatten_mlvl_points)
        flatten_mlvl_strides = torch.cat(flatten_mlvl_strides)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_id_scores = torch.cat(mlvl_id_scores)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        padding = mlvl_id_scores.new_zeros(mlvl_id_scores.shape[0], 1)
        mlvl_id_scores = torch.cat([mlvl_id_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        id_score_thr = cfg.id_score_thr
        cls_score_thr = cfg.cls_score_thr
        det_bboxes, det_kernels_pred, det_points, det_strides, det_inds = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_id_scores,
            mlvl_kernels_pred,
            flatten_mlvl_points,
            flatten_mlvl_strides,
            cls_score_thr,
            id_score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness,
            is_first=is_first)

        det_id_scores = mlvl_id_scores[det_inds]
        det_cls_scores = mlvl_scores[det_inds] * mlvl_centerness[det_inds].unsqueeze(1)

        # generate masks
        masks = None
        ori_masks = None
        if det_bboxes.shape[0] > 0:
            mask_head_params = det_kernels_pred
            num_instance = len(det_points)
            mask_head_inputs = self.relative_coordinate_feature_generator(
                mask_feat,
                det_points,
                det_strides)
            weights, biases = parse_dynamic_params(
                mask_head_params,
                8,
                self.weight_nums,
                self.bias_nums)
            mask_logits = self.mask_heads_forward(
                mask_head_inputs,
                weights,
                biases,
                num_instance)
            mask_logits = mask_logits.reshape(-1, 1, mask_feat.size(1), mask_feat.size(2))
            mask_logits = aligned_bilinear(mask_logits, 2).sigmoid()

            pred_global_masks = aligned_bilinear(mask_logits, 4)
            pred_global_masks = pred_global_masks[:, :, :img_shape[0], :img_shape[1]]
            ori_masks = F.interpolate(
                pred_global_masks,
                size=(ori_shape[0], ori_shape[1]),
                mode='bilinear',
                align_corners=False).squeeze(1)
            ori_masks.gt_(0.5)
            masks = aligned_bilinear(mask_logits, 4).squeeze(1)
            masks.gt_(0.5)
        return det_bboxes, det_id_scores, det_cls_scores, ori_masks, masks

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, gt_ids_list, gt_masks_list):
        """Get targets for points from ground truth annotations
           Not available for evaluation code
        """

        return None

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_strides = []
        for i in range(len(featmap_sizes)):
            points, strides = self.get_points_single(
                featmap_sizes[i],
                self.strides[i],
                dtype,
                device)
            mlvl_points.append(points)
            mlvl_strides.append(strides)

        return mlvl_points, mlvl_strides

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        strides = points[:,0] * 0 + stride
        return points, strides

    def simple_test(self, x, img_metas, rescale=False, is_first=False):
        """IAICondInstHead test without augmentation.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            rescale : whether to resize results to original scale

            is_end : whether this frame is the last image of the whole dataset

        Returns:
            bbox_results : a dict of bbox results

            segm_results : a dict of segmentation results

            id_masks : one hot ID masks, is used to create ID embedding

            new_inst_exists :  whether a new instance exists in current frame, is used to update global memory

            return_cls_scores : an ensemble dict of classification category & score for each instance in the video
        """
        outputs = self(x)
        bbox_inputs = outputs + (img_metas, self.test_cfg, rescale)

        bbox_results, segm_results, id_masks, new_inst_exists, cls_scores_dict = self.get_bboxes(*bbox_inputs, is_first=is_first)

        return [(bbox_results, segm_results)], id_masks, new_inst_exists, cls_scores_dict
