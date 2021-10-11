import sys
import re

from collections import Counter, defaultdict
from argparse import ArgumentParser
from transformers import AutoTokenizer


BASIC_TOKENIZE_RE = re.compile(r'([^\W_]+|.)')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('tokenizer')
    ap.add_argument('text')
    return ap


def basic_tokenize(text):
    tokens, offsets, i = [], [], 0
    for t in BASIC_TOKENIZE_RE.split(text):
        if t and not t.isspace():
            tokens.append(t)
            offsets.append((i, i+len(t)))
        i += len(t)
    return tokens, offsets


def normalize(text):
    text = text.replace('\xad', '')    # soft hyphen
    return text


def strip_space_from_offsets(text, offsets):
    stripped = []
    for s, e in offsets:
        while s < e and text[s].isspace():
            s += 1
        while s < e and text[e-1].isspace():
            e -= 1
        stripped.append((s, e))
    return stripped


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    basic_count, tokenized_count, unk_count = 0, 0, 0
    unk_counts, split_counts = Counter(), Counter()
    with open(args.text) as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip('\n')
            line = normalize(line)
            basic_tokens, basic_offsets = basic_tokenize(line)
            tokens = tokenizer.tokenize(line)
            t = tokenizer(line, return_offsets_mapping=True)
            tokenizer_offsets = strip_space_from_offsets(line, t['offset_mapping'])
            token_index = defaultdict(int)
            for i, (s, e) in enumerate(tokenizer_offsets, start=1):
                for j in range(s, e):
                    token_index[j] = i
            unks = Counter([
                line[s:e]
                for i, (s, e) in zip(t['input_ids'], tokenizer_offsets)
                if i == tokenizer.unk_token_id
            ])
            splits = Counter([
                line[s:e]
                for s, e in basic_offsets
                if token_index[s] != token_index[e-1]
                and line[s:e] not in unks
            ])
            basic_count += len(basic_tokens)
            tokenized_count += len(tokens)
            unk_count += sum(unks.values())
            unk_counts.update(unks)
            split_counts.update(splits)
    print('most common unknown basic tokens:', unk_counts.most_common(10))
    print('most common split basic tokens:', split_counts.most_common(10))
    print(f'unk/basic token ratio: {unk_count/basic_count:.4f} '
          f'({unk_count}/{basic_count})')
    print(f'token/basic token ratio: {tokenized_count/basic_count:.2f} '
          f'({tokenized_count}/{basic_count})')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
