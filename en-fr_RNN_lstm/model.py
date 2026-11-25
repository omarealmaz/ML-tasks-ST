from tensorflow import keras
from tensorflow.keras import layers
import config as C


def build_models(src_v, tgt_v, emb=C.EMB_DIM, units=C.UNITS, drop=C.DROPOUT):
    # Encoder
    enc_in = keras.Input((None,), dtype='int32', name='enc_in')      #placeholder for enc input     # (B, T)
    x = layers.Embedding(src_v, emb, mask_zero=True, name='enc_emb')(enc_in) #converts token id to embeddings # (B, T, E)
    x = layers.Dropout(drop)(x)
    _, h, c = layers.LSTM(units, return_state=True, name='enc_lstm')(x) #pass emb tokens x into encoder lstm - puts output in h,c   # h, c: (B, U)

    # Decoder, training version (teacher forcing)
    dec_in = keras.Input((None,), dtype='int32', name='dec_in')           # (B, T)
    dec_emb = layers.Embedding(tgt_v, emb, mask_zero=True, name='dec_emb')
    dec_lstm = layers.LSTM(units, return_sequences=True, return_state=True, name='dec_lstm')
    dec_dense = layers.Dense(tgt_v, name='dec_out')
    y = layers.Dropout(drop)(dec_emb(dec_in))                             # (B, T, E)
    y, _, _ = dec_lstm(y, initial_state=[h, c])                           # (B, T, U)
    logits = dec_dense(y)   #puts all scores of 16k vocab in a the tensor # (B, T, Vt)

    train_model = keras.Model([enc_in, dec_in], logits)

    # Inference models: same layer objects, so same weights
    encoder_model = keras.Model(enc_in, [h, c])
    s_tok = keras.Input((1,), dtype='int32')                              # (B, 1) (128*1)
    s_h = keras.Input((units,))                                           # (B, U) (128*512)
    s_c = keras.Input((units,))         #initialize model inputs          # (B, U)
    z, nh, nc = dec_lstm(dec_emb(s_tok), initial_state=[s_h, s_c])        # (B, 1, U)
    step_model = keras.Model([s_tok, s_h, s_c], [dec_dense(z), nh, nc])   # (B, 1, Vt), (B, U), (B, U)
    return train_model, encoder_model, step_model