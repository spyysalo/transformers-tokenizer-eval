import sys
import re

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
    with open(args.text) as f:
        for ln, line in enumerate(f, start=1):
            line = line.rstrip('\n')            
            basic_count += len(basic_tokenize(line))
            tokenized_count += len(tokenizer.tokenize(line))
    print(f'token/basic token ratio: {tokenized_count/basic_count:.2f} '
          f'({tokenized_count}/{basic_count})')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
