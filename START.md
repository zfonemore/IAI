## Getting started with IAI

### Model zoo

#### YTVIS2019

|      Name    | Backbone | Pretrain | AP   | AP50 | AP75 | AR1  | AR10 | Model |
| -------------| -------- | -------- | ---- | ---- | ---- | ---- | ---- | ----- |
| IAI+CondInst |   R50    | [COCO](https://drive.google.com/file/d/15w9jpvK8I5GrHYKWI8VOnmkc_gBU7aa2/view?usp=sharing) | 38.6 | 60.1 | 41.9 | 38.4 | 45.6 | [gdrive](https://drive.google.com/file/d/1v6DJKjoiBvwO0jAR3fNTLfnpAP4ZaEh8/view?usp=sharing) |
| IAI+CondInst |   R101   | [COCO](https://drive.google.com/file/d/1Tfg__rlo9VlMQWtIHPqvHzFwASPVb3U-/view?usp=sharing) | 41.9 | 63.7 | 47.5 | 41.1 | 49.6 | [gdrive](https://drive.google.com/file/d/18tKT_b37CPaZL6AMaA5_sfOSzTnNxzsk/view?usp=sharing) |


### Training

To train IAI+CondInst r50 on YouTube-VIS 2019 with 2 GPUs, run:

```
bash tools/dist_train.sh configs/iai/ytvis2019_iai_condinst_r50.py 2
```

To train IAI+CondInst r101 on YouTube-VIS 2019 with 2 GPUs, run:

```
bash tools/dist_train.sh configs/iai/ytvis2019_iai_condinst_r101.py 2
```

To train IAI+CondInst on YouTube-VIS 2021 with 2 GPUs, run:

```
bash tools/dist_train.sh configs/iai/ytvis2021_iai_condinst_r50.py 2
```

To train IAI+CondInst on OVIS with 2 GPUs, run:

```
bash tools/dist_train.sh configs/iai/ovis_iai_condinst_r50.py 2
```

### Inference & Evaluation


Evaluating on YouTube-VIS 2019 with single GPU, run:

```
python tools/test.py configs/iai/ytvis2019_iai_condinst_r50.py models/iai_condinst_r50.pth --eval segm
```

Evaluating on YouTube-VIS 2019 with multiple GPUs (e.g. 2 GPUs), run:

```
bash tools/dist_test.sh configs/iai/ytvis2019_iai_condinst_r50.py models/iai_condinst_r50.pth 2 
```

A json file with the predicted result will be generated as ```output/results.json```. To get validataion results, please zip the json file and upload it to the codalab server for [YouTube-VIS 2019](https://competitions.codalab.org/competitions/20128#participate-submit_results), [YouTube-VIS 2021](https://competitions.codalab.org/competitions/28988#participate-submit_results) and [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763).
