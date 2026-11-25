import sys
from translate import Translator

tr = Translator(sys.argv[1] if len(sys.argv) > 1 else 'main')
print('Type an English sentence (empty line to quit).')
while True:
    s = input('EN > ').strip()
    if not s:
        break
    print('greedy:', tr.greedy(s))
    print('beam  :', tr.beam(s))