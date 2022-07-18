PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/test.py configs/iai/iai_condinst_r101.py models/iai_condinst_r101.pth --eval segm
