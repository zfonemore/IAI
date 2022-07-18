import torch
import torch.nn.functional as F
from mmcv.ops.nms import batched_nms


def multiclass_nms(multi_bboxes,
                   multi_cls_scores,
                   multi_id_scores,
                   multi_kernels,
                   multi_points,
                   multi_strides,
                   cls_score_thr,
                   id_score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   is_first=False):
    bboxes = multi_bboxes.reshape(-1, 4)
    cls_scores = multi_cls_scores.max(dim=1)[0].reshape(-1)
    id_scores = multi_id_scores.max(dim=1)[0].reshape(-1)
    kernels = multi_kernels.reshape(-1, 169)
    points = multi_points.reshape(-1, 2)
    strides = multi_strides.reshape(-1, 1)

    # combine id_score & cls_score to remove low scoring boxes
    valid_mask = (cls_scores > cls_score_thr) & (id_scores > id_score_thr)
    # use combination of id_score & cls_score to rank points during nms
    # in first frame, cls_scores weighs more than id_score as no previous information
    if is_first:
        scores = cls_scores + 0.5 * id_scores
    else:
        scores = 0.5 * cls_scores + id_scores
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    cls_inds = (cls_scores > cls_score_thr).nonzero(as_tuple=False).squeeze(1)
    id_inds = (id_scores > id_score_thr).nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, kernels, points, strides = \
        bboxes[inds], scores[inds], kernels[inds], points[inds], strides[inds]
    return_inds = inds
    if inds.numel() == 0:
       return bboxes, kernels, points, strides, return_inds

    dets, keep = batched_nms(bboxes, scores, torch.ones(scores.shape), nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, kernels[keep], points[keep], strides[keep], return_inds[keep]

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)
    num_instances = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(
        torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if (l != num_layers - 1):
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances * channels)
        else:
            # out_channels x in_channels x 1 x 1 (out_channels = 1)
            weight_splits[l] = weight_splits[l].reshape(
                num_instances * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_instances)
    return weight_splits, bias_splits

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0,
        w * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shifts_y = torch.arange(0,
        h * stride,
        step=stride,
        dtype=torch.float32,
        device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor
    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor,
                           size=(oh, ow),
                           mode='bilinear',
                           align_corners=True)
    tensor = F.pad(tensor,
                   pad=(factor // 2, 0, factor // 2, 0),
                   mode="replicate")
    return tensor[:, :, :oh - 1, :ow - 1]


