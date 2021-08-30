# transformers-tokenizer-eval

Evaluation of transformers tokenizers

## Quickstart

Set up virtual environment

```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Run evaluation

```
python evaluate.py \
    TurkuNLP/bert-base-finnish-cased-v1 \
    example-data/fiwiki-sample.txt
```
