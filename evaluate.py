import sys
import re

from collections import Counter
from argparse import ArgumentParser
from transformers import AutoTokenizer


BASIC_TOKENIZE_RE = re.compile(r'([^\W_]+|.)')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('tokenizer')
    ap.add_argument('text')
    return ap


def basic_tokenize(text):
    return [
        t for t in BASIC_TOKENIZE_RE.split(text)
        if t and not t.isspace()
    ]


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    basic_count, tokenized_count = 0, 0
    unk_counts, split_counts = Counter(), Counter()
    with open(args.text) as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip('\n')
            basic_tokens = basic_tokenize(line)
            tokens = tokenizer.tokenize(line)
            t = tokenizer(line, return_offsets_mapping=True)
            unks = Counter([
                line[s:e]
                for i, (s, e) in zip(t['input_ids'], t['offset_mapping'])
                if i == tokenizer.unk_token_id
            ])
            splits = Counter({
                s: c
                for s, c in (Counter(basic_tokens)-Counter(tokens)).items()
                if s not in unks and len(s) > 1
            })
            basic_count += len(basic_tokens)
            tokenized_count += len(tokens)
            unk_counts.update(unks)
            split_counts.update(splits)
    print('most common unknown basic tokens:', unk_counts.most_common(10))
    print('most common split basic tokens:', split_counts.most_common(10))
    print(f'token/basic token ratio: {tokenized_count/basic_count:.2f} '
          f'({tokenized_count}/{basic_count})')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
