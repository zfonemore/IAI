## Getting Started with IAI

### Model zoo

Train on YouTube-VIS 2019, evaluate on YouTube-VIS 2019.

| Name  | pretrain | AP   | AP50 | AP75 | AR1  | AR10 | model |
| ----- | -------- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| R50   | [COCO](https://drive.google.com/file/d/15w9jpvK8I5GrHYKWI8VOnmkc_gBU7aa2/view?usp=sharing)     | 13.5 | 74.0 | 52.9 | 47.7 | 58.7 | [gdrive](https://drive.google.com/file/d/1v6DJKjoiBvwO0jAR3fNTLfnpAP4ZaEh8/view?usp=sharing) |
| R101  | [COCO](https://drive.google.com/file/d/1Tfg__rlo9VlMQWtIHPqvHzFwASPVb3U-/view?usp=sharing)     | 41.1 | 73.1 | 56.1 | 47.0 | 57.9 | [model](https://drive.google.com/file/d/18tKT_b37CPaZL6AMaA5_sfOSzTnNxzsk/view?usp=sharing) |


### Training

To train IAI on YouTube-VIS 2019 or OVIS with 2 GPUs , run:

```
sh tools/dist_train.sh --config-file projects/IDOL/configs/XXX.yaml 2 
```

### Inference & Evaluation



Evaluating on YouTube-VIS 2019 or OVIS:

```
python3 projects/IDOL/train_net.py --config-file projects/IDOL/configs/XXX.yaml --num-gpus 8 --eval-only
```



To get quantitative results, please zip the json file and upload to the [codalab server](https://competitions.codalab.org/competitions/20128#participate-submit_results) for YouTube-VIS 2019 and [server](https://codalab.lisn.upsaclay.fr/competitions/4763) for OVIS.


A json file with the predicted result will be generated as ```output/results.json```. YouTubeVIS currently only allows evaluation on the codalab server. Please upload the generated result to [codalab server](https://competitions.codalab.org/competitions/20128) to see performance.
