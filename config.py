from easydict import EasyDict as ED
from utils import ShapeSpec


cfg = ED()
# TODO: Most hyperparameters be hard code(so I can test it easy), it all will be added to this file later(maybe).


# trainer
# epoch, batch_size, so on.

# base info
cfg.NUM_CLASSES = 1
cfg.EPOCH = 100
cfg.lr = 1e-4

# data
cfg.ROOT = 'datasets/wgisd'
cfg.BATCH_SIZE = 3
cfg.NUM_WORKERS = 8
cfg.RESIZE = (800, 1280)

# backbone
# other backbone network's parameters will be added later, or ask me directly.
cfg.BACKBONE_DEPTH = 101
cfg.BACKBONE_OUTPUT_SHAPE = {
    'res2': ShapeSpec(channels=256, height=None, width=None, stride=4),
    'res3': ShapeSpec(channels=512, height=None, width=None, stride=8),
    'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16),
    'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)
}

# MASK
cfg.CLS_AGNOSTIC_MASK = False


