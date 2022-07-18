import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
import numpy as np
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def results2json_videoseg(dataset, results, out_file):
    json_results = []
    vid_objs = {}

    for idx in range(len(dataset)):
      vid_id, frame_id = dataset.img_ids[idx]
      if idx == len(dataset) - 1 :
        is_last = True
      else:
        _, frame_id_next = dataset.img_ids[idx+1]
        is_last = frame_id_next == 0
      det, seg = results[idx]
      for obj_id in det:
        bbox = det[obj_id]['bbox']
        segm = seg[obj_id]
        label = det[obj_id]['label']
        if obj_id not in vid_objs:
            vid_objs[obj_id] = {'scores':[],'cats':[], 'segms':{}}
        vid_objs[obj_id]['scores'].append(bbox[4])
        vid_objs[obj_id]['cats'].append(label)
        segm['counts'] = segm['counts'].decode()
        vid_objs[obj_id]['segms'][frame_id] = segm
      if is_last:
        # store results of  the current video
        for obj_id, obj in vid_objs.items():
          data = dict()

          data['video_id'] = vid_id + 1
          data['score'] = np.array(obj['scores']).mean().item()
          # majority voting for sequence category
          data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item() + 1
          vid_seg = []
          for fid in range(frame_id + 1):
            if fid in obj['segms']:
              vid_seg.append(obj['segms'][fid])
            else:
              vid_seg.append(None)
          data['segmentations'] = vid_seg
          json_results.append(data)
        vid_objs = {}
    if not osp.exists('./output'):
        import os
        os.mkdir('./output')
    mmcv.dump(json_results, out_file)

def manage_video_instance(results, scores_dict, first_frame, last_frame, new_results):
    ''' manage instance category & confidence score in a video
        average condidence scores across frames & get valid categories through the video
    '''
    new_id = 0
    new_ids_dict = {}
    new_scores_dict = {}
    new_labels_dict = {}
    for id, cls_scores in scores_dict.items():
        scores, labels = torch.topk(cls_scores, 5)
        valid_idx = (scores >= 0.05) & (labels != 40)
        new_scores_dict[id] = scores[valid_idx]
        new_labels_dict[id] = labels[valid_idx]
        new_ids_dict[id] = torch.arange(new_id, new_id+sum(valid_idx))
        new_id += sum(valid_idx)

    for j in range(first_frame, last_frame):
        new_result = {}
        new_bbox_results = {}
        new_segm_results = {}
        for id in results[j][0].keys():
            for label, score, new_id in zip(new_labels_dict[id], new_scores_dict[id], new_ids_dict[id]):
                new_bbox_result = {}
                new_bbox_result['bbox'] = np.append(results[j][0][id]['bbox'][:4], score.item())
                new_bbox_result['label'] = label.item()
                import copy
                new_segm_result = copy.deepcopy(results[j][1][id])
                new_bbox_results[new_id.item()] = new_bbox_result
                new_segm_results[new_id.item()] = new_segm_result
        new_result = [(new_bbox_results, new_segm_results)]

        new_results.extend(new_result)

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    curr_video_first_frame = 0
    new_results = []
    for i, data in enumerate(data_loader):
        is_end = (i == len(dataset)-1)
        with torch.no_grad():
            result, scores_dict = model(return_loss=False, rescale=True, is_end=is_end, **data)

        img_metas = data['img_metas'][0].data[0]
        if scores_dict is not None:
            if img_metas[0]['is_first']:
                curr_video_last_frame = i
                manage_video_instance(results, scores_dict, curr_video_first_frame, curr_video_last_frame, new_results)
                curr_video_first_frame = i
        batch_size = len(result)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                  for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()

    if scores_dict is not None:
        curr_video_last_frame = len(dataset)
        manage_video_instance(results, scores_dict, curr_video_first_frame, curr_video_last_frame, new_results)

    results2json_videoseg(dataset, new_results, './output/results.json')
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
