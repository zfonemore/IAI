import torch
import torch.nn.functional as F

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

def one_hot_mask(gt_masks, gt_ids_list, max_obj_num, bg=False):

    cls_num = 0
    num = 0
    n, h, w = gt_masks.shape
    new_masks = gt_masks.new_zeros(len(gt_ids_list), max_obj_num+2, h, w)
    #new_masks[:, max_obj_num+1, :, :] = 1

    if bg:
        return new_masks

    new_masks[:, max_obj_num+1, :, :] = 1
    for i, gt_ids in enumerate(gt_ids_list):
        for gt_id in gt_ids:
            new_masks[i][gt_id] = gt_masks[num]
            new_masks[i][max_obj_num+1][gt_masks[num].bool()] = 0
            num += 1
    return new_masks.contiguous()

def get_new_masks(gt_masks, gt_ori_ids, device, max_obj_num, one_hot=True, gt_labels=None):
    id_masks = []
    cls_masks = []
    INF = 1e12
    if gt_labels is not None:
        for gt_id, gt_label, gt_mask in zip(gt_ori_ids, gt_labels, gt_masks):
            try:
                gt_mask_tensor = gt_mask.to_tensor(dtype=torch.float32, device=device)
                h, w = gt_mask_tensor.size()[-2:]
                if len(gt_mask_tensor) == 0:
                    inds = gt_mask_tensor.new_zeros(h*w, dtype=torch.long)
                    per_im_id_masks = inds
                    per_im_id_masks[:] = max_obj_num + 1
                    per_im_cls_masks = inds
                    per_im_cls_masks[:] = 40
                else:
                    areas = gt_mask_tensor.sum(dim=-1).sum(dim=-1)
                    areas = areas[:, None, None].repeat(1, h, w)
                    areas[gt_mask_tensor == 0] = INF
                    areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                    min_areas, inds = areas.min(dim=1)
                    per_im_id_masks = gt_id[inds]
                    per_im_id_masks[min_areas == INF] = max_obj_num+1
                    per_im_cls_masks = gt_label[inds]
                    per_im_cls_masks[min_areas == INF] = 40
                per_im_id_masks = per_im_id_masks.reshape(h, w)
                per_im_cls_masks = per_im_cls_masks.reshape(h, w)
                id_masks.append(per_im_id_masks)
                cls_masks.append(per_im_cls_masks)
            except:
                import pdb
                pdb.set_trace()


    else:
        gt_masks_list = []
        for gt_id, gt_mask in zip(gt_ori_ids, gt_masks):
            try:
                gt_mask_tensor = gt_mask.to_tensor(dtype=torch.float32, device=device)
                gt_masks_list.append(gt_mask_tensor)
                h, w = gt_mask_tensor.size()[-2:]
                if len(gt_mask_tensor) == 0:
                    inds = gt_mask_tensor.new_zeros(h*w, dtype=torch.long)
                    per_im_id_masks = inds
                    per_im_id_masks[:] = max_obj_num + 1
                else:
                    areas = gt_mask_tensor.sum(dim=-1).sum(dim=-1)
                    areas = areas[:, None, None].repeat(1, h, w)
                    areas[gt_mask_tensor == 0] = INF
                    areas = areas.permute(1, 2, 0).reshape(h * w, -1)
                    min_areas, inds = areas.min(dim=1)
                    per_im_id_masks = gt_id[inds]
                    per_im_id_masks[min_areas == INF] = max_obj_num+1
                per_im_id_masks = per_im_id_masks.reshape(h, w)
                id_masks.append(per_im_id_masks)
            except:
                import pdb
                pdb.set_trace()

    id_masks = pad_mask(id_masks, max_obj_num+1)
    '''
    for i in range(len(id_masks)):
        pad_sum = (id_masks[i] == 0).sum()
        ori_sum = gt_masks_list[i][0].sum()
        if (pad_sum > 0) and (pad_sum != ori_sum):
            print('fuck')
            import pdb
            pdb.set_trace()
        #print('pad_sum:', (id_masks[i] == 0).sum())
        #print('ori_sum:', gt_masks_list[i][0].sum())
    '''
    id_masks = torch.stack(id_masks, dim=0)
    id_masks = id_masks.unsqueeze(1)
    if gt_labels is not None:
        cls_masks = pad_mask(cls_masks, 40)
        cls_masks = torch.stack(cls_masks, dim=0)
        cls_masks = cls_masks.unsqueeze(1)
    if one_hot:
        num_classes = max_obj_num + 2
        class_range = torch.arange(
            num_classes, dtype=torch.float32,
            device=device
        )[:, None, None]
        ori_offline_one_hot_masks = (id_masks == class_range).float()
        if gt_labels is not None:
            class_range = torch.arange(
                41, dtype=torch.float32,
                device=device
            )[:, None, None]
            cls_one_hot_masks = (cls_masks == class_range).float()
    else:
        ori_offline_one_hot_masks = id_masks

    bg_one_hot_masks = id_masks.new_zeros(ori_offline_one_hot_masks.shape, dtype=torch.float32)

    if gt_labels is not None:
        bg_cls_one_hot_masks = id_masks.new_zeros(cls_one_hot_masks.shape, dtype=torch.float32)
        return bg_one_hot_masks, ori_offline_one_hot_masks, id_masks, bg_cls_one_hot_masks, cls_one_hot_masks
    else:
        return bg_one_hot_masks, ori_offline_one_hot_masks, id_masks

def process_id(gt_ids, batch_size, max_obj_num):
    import copy
    ori_gt_ids = copy.deepcopy(gt_ids)
    for img_id in range(batch_size):
        ids_set = set()
        for frame in range(5):
            gt_id_set = set(gt_ids[img_id + frame * batch_size].cpu().numpy().tolist())
            new_ids = gt_id_set - ids_set
            if len(new_ids) > 0:
                for new_id in new_ids:
                    for index, gt_id in enumerate(gt_id_set):
                        if new_id == gt_id:
                            # set new id to gt id
                            #gt_ids[img_id + frame * batch_size][index] = max_obj_num + 1
                            gt_ids[img_id + frame * batch_size][index] = max_obj_num
            ids_set = ids_set | gt_id_set

    return ori_gt_ids

def pad_mask(masks, max_obj_num):

    max_shape = [0 for dim in range(2)]
    for mask in masks:
        shapes = mask.size()[-2:]
        for dim in range(2):
            max_shape[dim] = max(max_shape[dim], shapes[-dim-1])
    pad = [0 for dim in range(4)]
    padded_mask = []
    for mask in masks:
        shapes = mask.size()[-2:]
        for dim in range(2):
            pad[2*dim+1] = max_shape[dim] - shapes[-dim-1]
        if max(pad) > 0:
            padded_mask.append(F.pad(mask, pad, value=0))
        else:
            padded_mask.append(mask)

    return padded_mask

def split_frames(xs, chunk_size):
    new_xs = []
    for x in xs:
        all_x = list(torch.split(x, chunk_size, dim=0))
        new_xs.append(all_x)
    return list(zip(*new_xs))
