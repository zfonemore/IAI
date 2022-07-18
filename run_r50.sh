PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py configs/iai/iai_condinst_r50.py models/iai_condinst_r50.pth --eval segm
