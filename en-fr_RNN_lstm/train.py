import argparse
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras

import config as C
from data import load_pairs, get_tokenizers, encode, make_ds
from model import build_models

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='main')
ap.add_argument('--n', type=int, default=None, help='use only n training pairs')
ap.add_argument('--epochs', type=int, default=C.EPOCHS)
ap.add_argument('--batch', type=int, default=C.BATCH)
ap.add_argument('--units', type=int, default=C.UNITS)
ap.add_argument('--vocab', type=int, default=C.VOCAB_SIZE)
ap.add_argument('--drop', type=float, default=C.DROPOUT)
ap.add_argument('--overfit', action='store_true', help='sanity test: no dropout, no early stopping')
args = ap.parse_args()

random.seed(C.SEED)
np.random.seed(C.SEED)
tf.random.set_seed(C.SEED)
P = C.paths(args.run)
drop = 0.0 if args.overfit else args.drop

train_pairs = load_pairs('train', args.n) # from data.py
val_pairs = load_pairs('validation')

en_tok, fr_tok = get_tokenizers(train_pairs, P, args.vocab) #tokenize

tr = encode(train_pairs, en_tok, fr_tok) #encode train pairs - returns enc_in , dec_in , dec_out
va = encode(val_pairs, en_tok, fr_tok) #encode validation pairs

print('train arrays', [a.shape for a in tr], 'val arrays', [a.shape for a in va])

train_ds = make_ds(tr, args.batch)
val_ds = make_ds(va, C.BATCH, shuffle=False)

model, _, _ = build_models(en_tok.get_vocab_size(), fr_tok.get_vocab_size(),
                           units=args.units, drop=drop)
model.summary()


model.compile(
    optimizer=keras.optimizers.Adam(C.LR, clipnorm=1.0),

    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0),
)

monitor = 'loss' if args.overfit else 'val_loss'

callbacks = [#automated steps that runs after each epoch
    keras.callbacks.ModelCheckpoint(P['weights'], save_best_only=True,
                                    save_weights_only=True, monitor=monitor), #saves lowest val loss epoch
    keras.callbacks.CSVLogger(P['log']),
]

if not args.overfit:
    callbacks += [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1),
    ]

hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks) #model execution

with open(P['hparams'], 'w') as f:
    json.dump({'emb': C.EMB_DIM, 'units': args.units, 'drop': drop}, f)

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss (cross-entropy per token)')
plt.legend()
plt.savefig(P['plot'], dpi=150)
print('saved to', P['dir'])