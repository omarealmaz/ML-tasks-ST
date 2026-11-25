import argparse
import random
import sacrebleu
from tqdm import tqdm

import config as C
from data import load_pairs
from translate import Translator

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='main')
ap.add_argument('--split', default='test')
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--k', type=int, default=4)
ap.add_argument('--max-src-len', type=int, default=0,
                help='keep only sentences with at most this many English tokens (0 = all)')
ap.add_argument('--no-beam', action='store_true')
args = ap.parse_args()

random.seed(C.SEED)
tr = Translator(args.run)
pairs = load_pairs(args.split)
if args.max_src_len:
    pairs = [p for p in pairs if len(tr.en_tok.encode(p[0]).ids) <= args.max_src_len]
pairs = random.sample(pairs, min(args.n, len(pairs)))
refs = [[fr for _, fr in pairs]]
print(f'evaluating {len(pairs)} sentences from the {args.split} split')

systems = [('greedy', tr.greedy)]
if not args.no_beam:
    systems.append((f'beam k={args.k}', lambda s: tr.beam(s, k=args.k)))

for name, fn in systems:
    hyps = [fn(en) for en, _ in tqdm(pairs, desc=name)]
    print(f'{name}: BLEU {sacrebleu.corpus_bleu(hyps, refs).score:.1f}'
          f'   chrF {sacrebleu.corpus_chrf(hyps, refs).score:.1f}')
    for i in range(5):
        print('  EN :', pairs[i][0])
        print('  REF:', pairs[i][1])
        print('  HYP:', hyps[i])