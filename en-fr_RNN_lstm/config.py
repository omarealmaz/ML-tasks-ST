import os

MAX_LEN = 40
VOCAB_SIZE = 16000
EMB_DIM = 256
UNITS = 512
DROPOUT = 0.2
BATCH = 128
EPOCHS = 20
LR = 1e-3
SEED = 42
PAD, UNK, START, END = '[PAD]', '[UNK]', '[START]', '[END]'


def paths(run='main'):
    d = os.path.join('runs', run)
    os.makedirs(d, exist_ok=True)
    return {
        'dir': d,
        'en_tok': os.path.join(d, 'en_tok.json'),
        'fr_tok': os.path.join(d, 'fr_tok.json'),
        'weights': os.path.join(d, 'best.weights.h5'),
        'hparams': os.path.join(d, 'hparams.json'),
        'log': os.path.join(d, 'train_log.csv'),
        'plot': os.path.join(d, 'loss_curve.png'),
    }