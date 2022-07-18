import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2roi
from ..builder import DETECTORS, build_head, build_neck, build_roi_extractor, build_loss
from .single_stage import SingleStageDetector
from mmcv.cnn import kaiming_init
from fvcore.nn import sigmoid_focal_loss_jit


@DETECTORS.register_module()
class IAICondInst(SingleStageDetector):
    """IAI paradigm on condinst detectors for VIS.
        Add lstt block to associate features
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 lstt_block=None,
                 bbox_roi_extractor=None,
                 id_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(IAICondInst, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.num_frames = id_cfg.num_frames
        self.batch_size = id_cfg.batch_size
        self.max_obj_num = id_cfg.max_obj_num
        self.new_inst_exist = False

        # lstt block for implementing association module
        self.lstt = build_head(lstt_block)

        # project backbone features for lstt block to encode ID features
        self.encoder_projector = nn.Conv2d(
            2048, 256, kernel_size=1)
        # project backbone features for preserving classification features
        self.backbone_projector = nn.Conv2d(
            2048, 1536, kernel_size=1)
        self.cls_scores = {}

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        x = self.backbone(img)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_ids=None,
                      proposals=None):
        """training forward of IAICondInst model
        Not available for evaluation code
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        pass

        return None

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, is_end=False):
        """Test without augmentation.
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

            rescale : whether to resize results to original scale

            is_end : whether this frame is the last image of the whole dataset

        Returns:
            results : (bbox results, segmentation results)

            return_cls_scores : an ensemble dict of classification category & score for each instance in the video
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        feats = self.extract_feat(img)
        is_first = img_metas[0]['is_first']

        if is_first:
            # average classification scores for the last video
            import copy
            return_cls_scores = copy.deepcopy(self.cls_scores)
            for key, val in return_cls_scores.items():
                return_cls_scores[key] = val / self.cls_scores_num[key]
            self.cls_scores = {}
            self.cls_scores_num = {}
            self.lstt.restart(batch_size=1, enable_id_shuffle=False)
            h, w = img.size()[2:]
            prev_one_hot_masks = img.new_zeros(1, self.max_obj_num+1, h, w)
        else:
            return_cls_scores = None
            prev_one_hot_masks = self.pred_masks

        # use lstt to combine backbone features & ID embedding
        new_feat = self.encoder_projector(feats[-1])
        new_feats = (feats[0], feats[1], feats[2], new_feat)
        lstt_embs = self.lstt(new_feats, prev_one_hot_masks, self.new_inst_exist)
        embs = [new_feats[-1]]
        n, c, h, w = new_feats[-1].size()
        for emb in lstt_embs:
            embs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))
        backbone_cls_feat = self.backbone_projector(feats[-1])
        embs.append(backbone_cls_feat)
        embs = torch.cat(embs, dim=1)
        lstt_feats = (feats[0], feats[1], feats[2], embs)

        enc_feats = self.neck(lstt_feats)

        results, self.pred_masks, self.new_inst_exist, cls_scores = \
            self.bbox_head.simple_test(enc_feats, img_metas, rescale=rescale,
                                    is_first=is_first)

        # update local memory & global memory
        # for first frame there is not previous memory, so reset memory
        if is_first:
            self.lstt.reset_memory(self.pred_masks)
        else:
            self.lstt.update_short_term_memory(self.pred_masks, self.new_inst_exist)

        if cls_scores is not None:
            for obj_id, cls_score in cls_scores.items():
                if obj_id in self.cls_scores:
                    self.cls_scores[obj_id] += cls_score
                    self.cls_scores_num[obj_id] += 1
                else:
                    self.cls_scores[obj_id] = cls_score
                    self.cls_scores_num[obj_id] = 1

        # if is the last image of the whole dataset, average classification scores for the current video
        if is_end:
            import copy
            return_cls_scores = copy.deepcopy(self.cls_scores)
            for key, val in return_cls_scores.items():
                return_cls_scores[key] = val / self.cls_scores_num[key]

        return results, return_cls_scores
