## Prepareing Dataset
You can download VIS dataset from the CodaLab evaluation server of [YouTubeVIS2019](https://competitions.codalab.org/competitions/20128#participate-get-data), [YouTubeVIS2021](https://competitions.codalab.org/competitions/28988#participate) and [OVIS](https://codalab.lisn.upsaclay.fr/competitions/4763#participate).  

## Expected dataset path
Extract the VIS dataset and put them under data path (or link dataset to this path).

```
IAI
├── data
│   ├──ytvis2019
│   ├──ytvis2021
│   ├──ovis
```

## Expected dataset structure for YTVIS2019

```
ytvis2019
├── train
├── val
├── annotations
│   ├── instances_train_sub.json
│   ├── instances_val_sub.json
```

## Expected dataset structure for YTVIS2021

```
ytvis2021
├── train
├── val
```

## Expected dataset structure for OVIS

```
ovis
├── train
├── valid
├── annotations_train.json
├── annotations_valid.json
```


