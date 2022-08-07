## Installation

### Requirements

- Python3
- Pytorch >= 1.5
- MMCV 1.3.8
- pycocotools-VIS
- Pytorch Correlation (Recommend to install from [source](https://github.com/ClementPinard/Pytorch-Correlation-extension) instead of using pip.)

### Install Command

Install pycocotools-VIS
```
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

Install MMCV

When installing mmcv, you can set [cu_version} to cu92 or other version, {torch_version} to 1.5.0 or other version.
For example, if your environment is cuda 9.2 & torch 1.5.0, use 
```
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu92/1.5.0/index.html
```

Install IAI
```
cd IAI
pip install -v -e .
```
