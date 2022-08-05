CONFIG_PATH=$1
MODEL_PATH=$2
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python tools/test.py $CONFIG_PATH $MODEL_PATH --eval segm
