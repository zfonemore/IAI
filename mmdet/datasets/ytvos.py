import numpy as np
import os.path as osp
import random
import mmcv
from .custom import CustomDataset
import torch
from .pycoco_ytvos import YTVOS
from mmcv.parallel import DataContainer as DC
from .builder import DATASETS
from collections import Sequence
from .pipelines import Compose


@DATASETS.register_module()
class YTVOSDataset(CustomDataset):

    CLASSES=('person','giant_panda','lizard','parrot','skateboard','sedan',
        'ape','dog','snake','monkey','hand','rabbit','duck','cat','cow','fish',
        'train','horse','turtle','bear','motorbike','giraffe','leopard',
        'fox','deer','owl','surfboard','airplane','truck','zebra','tiger',
        'elephant','snowboard','boat','shark','mouse','frog','eagle','earless_seal',
        'tennis_racket')

    YTVIS2021_CLASSES=('airplane', 'bear', 'bird', 'boat', 'car', 'cat', 'cow',
        'deer', 'dog', 'duck', 'earless_seal', 'elephant', 'fish',
        'flying_disc', 'fox', 'frog', 'giant_panda', 'giraffe',
        'horse', 'leopard', 'lizard', "monkey", "motorbike", "mouse",
        "parrot", "person", "rabbit", "shark", "skateboard", "snake",
        "snowboard", "squirrel", "surfboard", "tennis_racket", "tiger",
        "train", "truck", "turtle", "whale", "zebra")

    def __init__(self,
                 ann_file,
                 img_prefix,
                 seg_prefix=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 with_track=False,
                 extra_aug=None,
                 aug_ref_bbox_param=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 pipeline=None):
        # prefix of images path
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix

        self.total_frames = 0
        # load annotations (and proposals)
        img_ids = []
        self.vid_infos = self.load_annotations(ann_file)
        for idx, vid_info in enumerate(self.vid_infos):
            if test_mode:
                img_ids_pervideo = []
            for frame_id in range(len(vid_info['filenames'])):
                if test_mode:
                    img_ids_pervideo.append((idx, frame_id))
                else:
                    img_ids.append((idx, frame_id))
                self.total_frames += 1
            if test_mode:
                img_ids.append(img_ids_pervideo)

        self.img_ids = img_ids
        self.proposal_file = proposal_file
        if proposal_file is not None:
            self.proposals = self.load_proposals(proposal_file)
        else:
            self.proposals = None

        # filter images with no annotation during training
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]
            self.vid_ids = []
            for idx, vid_info in enumerate(self.vid_infos):
                valid_inds = [i for i in range(len(vid_info['filenames']))
                                if len(self.get_ann_info(idx, i)['bboxes'])]
                self.vid_ids.append(valid_inds)

        # max proposals per image
        self.num_max_proposals = num_max_proposals

        # with mask or not (reserved field, takes no effect)
        self.with_mask = with_mask
        # some datasets provide bbox annotations as ignore/crowd/difficult,
        # if `with_crowd` is True, then these info is returned.
        self.with_crowd = with_crowd
        # with label is False for RPN
        self.with_label = with_label
        self.with_track = with_track
        # params for augmenting bbox in the reference frame
        self.aug_ref_bbox_param = aug_ref_bbox_param
        # in test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio

        # processing pipeline
        self.pipeline = Compose(pipeline)

        # num_frames in a video
        self.num_frames = 5


    def __len__(self):

        if self.test_mode:
            return len(self.img_ids)
        else:
            return len(self.vid_infos) * 5

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(self.img_ids[idx])

        idx = idx % len(self.vid_infos)
        vid_info = self.vid_infos[idx]
        vid_len = len(self.vid_ids[idx])
        vid_frame_ids = self.vid_ids[idx]
        if self.num_frames > vid_len:
            new_vid_frame_ids = [i for i in vid_frame_ids]
            for i in range(vid_len, self.num_frames):
                new_vid_frame_ids.append(vid_frame_ids[i % vid_len])

            vid_len = self.num_frames
            vid_frame_ids = new_vid_frame_ids
        curr_frame_id = random.randint(0, vid_len - self.num_frames)
        frame_ids = [vid_frame_ids[curr_frame_id]]
        for i in range(self.num_frames-1, 0, -1):
            interval = random.randint(1,min(3, vid_len - curr_frame_id - i))
            curr_frame_id += interval
            frame_ids.append(vid_frame_ids[curr_frame_id])

        data_list = []
        obj_ids_list = []
        scale = None
        for frame_id in frame_ids:
            data = None
            while data is None:
                data, obj_ids, obj_nums = self.prepare_train_img((idx, frame_id), scale)
            if 'scale' in data.keys():
                scale = data['scale']
                data.pop('scale')

            obj_ids_list += obj_ids
            data_list.append(data)

        obj_ids_list = np.unique(np.array(obj_ids_list))
        map = {x: i for i, x in enumerate(obj_ids_list)}
        for data in data_list:
            data['gt_ids']._data.apply_(lambda x:map[x])

        return data_list

    def load_annotations(self, ann_file):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i # + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            vid_infos.append(info)
        return vid_infos

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def bbox_aug(self, bbox, img_size):
        assert self.aug_ref_bbox_param is not None
        center_off = self.aug_ref_bbox_param[0]
        size_perturb = self.aug_ref_bbox_param[1]

        n_bb = bbox.shape[0]
        # bbox center offset
        center_offs = (2*np.random.rand(n_bb, 2) - 1) * center_off
        # bbox resize ratios
        resize_ratios = (2*np.random.rand(n_bb, 2) - 1) * size_perturb + 1
        # bbox: x1, y1, x2, y2
        centers = (bbox[:,:2]+ bbox[:,2:])/2.
        sizes = bbox[:,2:] - bbox[:,:2]
        new_centers = centers + center_offs * sizes
        new_sizes = sizes * resize_ratios
        new_x1y1 = new_centers - new_sizes/2.
        new_x2y2 = new_centers + new_sizes/2.
        c_min = [0,0]
        c_max = [img_size[1], img_size[0]]
        new_x1y1 = np.clip(new_x1y1, c_min, c_max)
        new_x2y2 = np.clip(new_x2y2, c_min, c_max)
        bbox = np.hstack((new_x1y1,new_x2y2)).astype(np.float32)
        return bbox

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in sample_range:
          # check if the frame id is valid
          ref_idx = (vid, i)
          if i != frame_id and ref_idx in self.img_ids:
              valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def prepare_train_img(self, idx, scale=None):
        vid,  frame_id = idx
        vid_info = self.vid_infos[vid]

        # obj ids attribute does not exist in current annotation, need to add it
        ann = self.get_ann_info(vid, frame_id)
        obj_ids = ann['obj_ids']
        obj_nums = len(obj_ids)

        vid_info['filename'] = vid_info['filenames'][frame_id]

        results = dict(img_info=vid_info, ann_info=ann)

        self.pre_pipeline(results)
        # sync scale for multiple gpus training
        if scale is not None:
            results['scale'] = scale
        results_processed = self.pipeline(results)

        return results_processed, obj_ids, obj_nums

    def prepare_test_img(self, idx_pervideo):
        """Prepare an image for testing (multi-scale and flipping)"""
        results_pervideo = []
        vid, _ = idx_pervideo[0]
        vid_info = self.vid_infos[vid]
        for vid, frame_id in idx_pervideo:

            vid_info['filename'] = vid_info['filenames'][frame_id]

            results = dict(img_info=vid_info)

            results['vid'] = vid + 1
            self.pre_pipeline(results)
            results_processed = self.pipeline(results)
            results_pervideo.append(results_processed)

        return results_pervideo

    def _parse_ann_info(self, ann_info, frame_id, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_ids = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        for i, ann in enumerate(ann_info):
            # each ann is a list of masks
            # ann:
            # bbox: list of bboxes
            # segmentation: list of segmentation
            # category_id
            # area: list of area
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            #bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
                if with_mask:
                    #gt_masks.append(self.ytvos.annToMask(ann, frame_id))
                    gt_masks.append(segm)
                    if isinstance(segm, dict):
                        mask_polys = [
                            p for p in segm if len(p) >= 6
                        ]  # valid polygons have >= 3 points (6 coordinates)
                        poly_lens = [len(p) for p in mask_polys]
                    else:
                        mask_polys = segm
                        poly_lens = [len(segm)]
                        gt_mask_polys.append(mask_polys)
                    gt_poly_lens.extend(poly_lens)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, obj_ids=gt_ids, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        return None
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
