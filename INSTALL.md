## Installation
This repo is built on MMDetection-2.11, and it requires Python3 and PyTorch >= 1.5. This repo also relies on MMCV-1.3.8 & pycocotools-VIS-version, and you could install them by using following commands.
```
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"

pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
When installing mmcv, you can set [cu_version} to cu92 or other version, {torch_version} to 1.5.0 or other version.
For example, if your environment is cuda 9.2 & torch 1.5.0, use 
```
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu92/1.5.0/index.html
```


