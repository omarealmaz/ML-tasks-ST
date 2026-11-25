import numpy as np
import tensorflow as tf

keras = tf.keras

import config as C
from data import load_pairs, get_tokenizers, encode
from model import build_models

P = C.paths('test')
pairs = load_pairs('train', 500)
en_tok, fr_tok = get_tokenizers(pairs, P, 2000)
enc_in, dec_in, dec_out = encode(pairs, en_tok, fr_tok)
S, E = fr_tok.token_to_id(C.START), fr_tok.token_to_id(C.END)
Vs, Vt = en_tok.get_vocab_size(), fr_tok.get_vocab_size()

# 1. shapes and padding id
assert enc_in.shape == dec_in.shape == dec_out.shape and enc_in.shape[1] == C.MAX_LEN
assert en_tok.token_to_id(C.PAD) == 0 and fr_tok.token_to_id(C.PAD) == 0
print('1 ok: arrays', enc_in.shape)

# 2. decoder input is the target shifted by one
assert (dec_in[:, 0] == S).all()
for i in range(len(dec_in)):
    n = int((dec_out[i] != 0).sum())
    assert dec_out[i, n - 1] == E
    assert (dec_in[i, 1:n] == dec_out[i, :n - 1]).all()
print('2 ok: shift is correct')

# 3. tokenizer round trip (eyeball it)
print('EN :', en_tok.decode(enc_in[0][enc_in[0] != 0].tolist()))
print('FR :', fr_tok.decode(dec_out[0][dec_out[0] != 0].tolist()))

# 4. model output shapes
tm, enc_m, step_m = build_models(Vs, Vt)
x = {'enc_in': enc_in[:4], 'dec_in': dec_in[:4]}
print('4 logits', tm(x).shape, 'expected (4,', C.MAX_LEN, ',', Vt, ')')
h, c = enc_m(enc_in[:1])
print('  h, c', h.shape, c.shape, 'expected (1,', C.UNITS, ')')
lg, h2, c2 = step_m([tf.constant([[S]], dtype='int32'), h, c])
print('  step', lg.shape, h2.shape, 'expected (1, 1,', Vt, ') (1,', C.UNITS, ')')

# 5. padding mask check
tm.compile(optimizer='adam',
           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0))
xb, yb = {'enc_in': enc_in[:64], 'dec_in': dec_in[:64]}, dec_out[:64]
keras_loss = tm.evaluate(xb, yb, batch_size=64, verbose=0)
per_tok = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')(yb, tm(xb, training=False)).numpy()
m = (yb != 0).astype('float32')
print('5 keras loss        :', round(float(keras_loss), 4))
print('  real-token mean    :', round(float((per_tok * m).sum() / m.sum()), 4))
print('  mean incl. padding :', round(float(per_tok.mean()), 4))
print('  expected start loss: about', round(float(np.log(Vt)), 2))