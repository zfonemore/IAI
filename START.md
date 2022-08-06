## Getting started with IAI

### Model zoo

#### YTVIS2019

|      Name    | Backbone | Pretrain | AP   | AP50 | AP75 | AR1  | AR10 | Model |
| -------------| -------- | -------- | ---- | ---- | ---- | ---- | ---- | ----- |
| IAI+CondInst |   R50    | [COCO](https://drive.google.com/file/d/15w9jpvK8I5GrHYKWI8VOnmkc_gBU7aa2/view?usp=sharing) | 38.6 | 60.1 | 41.9 | 38.4 | 45.6 | [gdrive](https://drive.google.com/file/d/1v6DJKjoiBvwO0jAR3fNTLfnpAP4ZaEh8/view?usp=sharing) |
| IAI+CondInst |   R101   | [COCO](https://drive.google.com/file/d/1Tfg__rlo9VlMQWtIHPqvHzFwASPVb3U-/view?usp=sharing) | 41.9 | 63.7 | 47.5 | 41.1 | 49.6 | [gdrive](https://drive.google.com/file/d/18tKT_b37CPaZL6AMaA5_sfOSzTnNxzsk/view?usp=sharing) |

#### YTVIS2021

|      Name    | Backbone | Pretrain | AP   | AP50 | AP75 | AR1  | AR10 | Model |
| -------------| -------- | -------- | ---- | ---- | ---- | ---- | ---- | ----- |
| IAI+CondInst |   R50    | COCO | 37.7 | 58.0 | 42.3 | 34.6 | 45.6 | gdrive |

#### OVIS

|      Name    | Backbone | Pretrain | AP   | AP50 | AP75 | AR1  | AR10 | Model |
| -------------| -------- | -------- | ---- | ---- | ---- | ---- | ---- | ----- |
| IAI+CondInst |   R50    | COCO+YTVIS2019 | 20.3 | 39.5 | 19.0 | 11.9 | 26.2 | gdrive |


### Training

Before training, please put the COCO pretrained model under the model folder. 

To train model with multple GPUs, run:
```
bash tools/dist_train.sh $CONFIG_PATH 2
```

For example, to train IAI+CondInst r50 on YouTube-VIS 2019 with 2 GPUs, run:

```
bash tools/dist_train.sh configs/iai/ytvis2019_iai_condinst_r50.py 2
```

After training, the model will be saved in the output folder.

### Inference & Evaluation

Evaluating on YouTube-VIS 2019 r50 with single GPU, run:

```
python tools/test.py $CONFIG_PATH $MODEL_PATH --eval segm
```

Evaluating on YouTube-VIS 2019 r50 with multiple GPUs (e.g. 2 GPUs), run:

```
bash tools/dist_test.sh $CONFIG_PATH $MODEL_PATH 2 
```

After evaluating, a json file with the predicted result will be generated as ```output/results.json```. To get validataion results, please zip the json file and upload it to the codalab server for [YouTube-VIS 2019](https://competitions.codalab.org/competitions/20128#participate-submit_results), [YouTube-VIS 2021](https://competitions.codalab.org/competitions/28988#participate-submit_results) and [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763).
