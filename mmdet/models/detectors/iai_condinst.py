import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2roi
from ..builder import DETECTORS, build_head, build_neck, build_roi_extractor, build_loss
from .single_stage import SingleStageDetector
from .utils import split_frames, process_id, get_new_masks, aligned_bilinear


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
        batch_size = self.batch_size
        x = self.extract_feat(img)
        feats_per_frame = split_frames(x, self.batch_size)
        gt_ori_ids = process_id(gt_ids, batch_size, self.max_obj_num-1)

        bg_one_hot_masks, ori_one_hot_masks, gt_masks_tensor = get_new_masks(gt_masks, gt_ori_ids, x[0].device, self.max_obj_num-1)

        lstt_embs = None
        losses = dict()
        rpn_losses_all = {}
        roi_losses_all = {}
        sem_losses_all = {}
        self.lstt.restart(batch_size=batch_size, enable_id_shuffle=False)
        prev_one_hot_masks = bg_one_hot_masks[:batch_size]
        bce_losses = []
        iou_losses = []

        for i, feats in enumerate(feats_per_frame):
            is_first = (i == 0)
            indices = slice(i*batch_size, (i+1)*batch_size)

            img_metas_per_frame = img_metas[indices]
            gt_bboxes_per_frame = gt_bboxes[indices]
            gt_labels_per_frame = gt_labels[indices]
            gt_masks_per_frame = gt_masks[indices]
            gt_ids_per_frame = gt_ids[indices]
            gt_ori_ids_per_frame = gt_ori_ids[indices]
            gt_masks_tensor_per_frame = gt_masks_tensor[indices]

            new_inst_exist=False
            for gt_id in gt_ids_per_frame:
                if self.max_obj_num in gt_id:
                    new_inst_exist=True

            new_feat = self.encoder_projector(feats[-1])
            new_feats = (feats[0], feats[1], feats[2], new_feat)

            lstt_embs = self.lstt(new_feats, prev_one_hot_masks, new_inst_exist)
            if lstt_embs is not None:
                embs = [new_feats[-1]]
                n, c, h, w = new_feats[-1].size()
                for emb in lstt_embs:
                    embs.append(emb.view(h, w, n, c).permute(2, 3, 0, 1))

                backbone_cls_feat = self.backbone_projector(feats[-1])
                embs.append(backbone_cls_feat)

                embs = torch.cat(embs, dim=1)
            lstt_feats = (feats[0], feats[1], feats[2], embs)
            enc_feats = self.neck(lstt_feats)

            roi_losses = self.bbox_head.forward_train(enc_feats,
                                                     img_metas_per_frame,
                                                     gt_bboxes_per_frame,
                                                     gt_labels_per_frame,
                                                     gt_bboxes_ignore, gt_masks_per_frame,
                                                     gt_ids_per_frame)
            for name, loss in roi_losses.items():
                if name not in roi_losses_all:
                    roi_losses_all[name] = [loss]
                else:
                    roi_losses_all[name].append(loss)

            prev_one_hot_masks = ori_one_hot_masks[indices]

            if is_first:
                self.lstt.reset_memory(prev_one_hot_masks)
            else:
                self.lstt.update_short_term_memory(prev_one_hot_masks, new_inst_exist)

        rpn_losses = {}
        for name, loss_list in rpn_losses_all.items():
            rpn_losses[name] = []
            for loss in loss_list:
                rpn_losses[name].append(sum(loss) / len(loss))
        losses.update(rpn_losses)

        roi_losses = {}
        for name, loss in roi_losses_all.items():
            roi_losses[name] = sum(loss) / len(loss)
        losses.update(roi_losses)

        return losses

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
        feats_video = self.extract_feat(img)
        feats_per_frame = split_frames(feats_video, 1)

        cls_scores_list = []
        pred_masks_list = []

        for frame_idx, feats in enumerate(feats_per_frame):
            is_first = (frame_idx == 0)

            if is_first:
                # average classification scores for the last video
                self.lstt.restart(batch_size=1, enable_id_shuffle=False)
                h, w = img.size()[2:]
                prev_one_hot_masks = img.new_zeros(1, self.max_obj_num+1, h, w)
            else:
                prev_one_hot_masks = id_masks

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

            pred_masks, id_masks, new_inst_exist, cls_scores = \
                self.bbox_head.simple_test(enc_feats, img_metas[0], rescale=rescale,
                                        is_first=is_first)

            cls_scores_list.append(cls_scores)
            pred_masks_list.append(pred_masks)
            # update local memory & global memory
            # In the first frame there is no previous memory, so reset memory
            if is_first:
                self.lstt.reset_memory(id_masks)
            else:
                self.lstt.update_short_term_memory(id_masks, new_inst_exist)

        results = []
        cls_scores_list = torch.cat(cls_scores_list)
        pred_masks_list = torch.cat(pred_masks_list)
        num_classes = cls_scores_list.shape[-1]

        cls_scores_mean = torch.mean(cls_scores_list, dim=0).flatten(0)
        cls_scores_topk, idx_topk = torch.topk(cls_scores_mean, 10)

        score_thr = 0.05
        keep = cls_scores_topk > score_thr
        idx_topk = idx_topk[keep]
        cls_scores_topk = cls_scores_topk[keep]

        obj_idx = idx_topk // num_classes
        pred_labels = idx_topk % num_classes

        pred_masks_list = pred_masks_list[:, obj_idx].transpose(0,1).bool().cpu().numpy()

        return cls_scores_topk, pred_labels, pred_masks_list
