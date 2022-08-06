## Getting Started with IAI
Download pretrained r50 model from [here](https://drive.google.com/file/d/1v6DJKjoiBvwO0jAR3fNTLfnpAP4ZaEh8/view?usp=sharing), and put it to `models/iai_condinst_r50.pth`

Run the following command to evaluate IAI + CondInst model with R50 backbone
```
sh run_r50.sh
```

Download pretrained r101 model from [here](https://drive.google.com/file/d/18tKT_b37CPaZL6AMaA5_sfOSzTnNxzsk/view?usp=sharing), and put it to `models/iai_condinst_r101.pth`

Run the following command to evaluate IAI + CondInst model with R101 backbone
```
sh run_r101.sh
```

A json file with the predicted result will be generated as ```output/results.json```. YouTubeVIS currently only allows evaluation on the codalab server. Please upload the generated result to [codalab server](https://competitions.codalab.org/competitions/20128) to see performance.
