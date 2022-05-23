import os

from transformers import BertTokenizer

from . import mkdir


if not os.path.exists('.cache/'):
    mkdir('.cache/')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.save_pretrained(save_directory='.cache')
else:
    tokenizer = BertTokenizer.from_pretrained('.cache/')

num_tokens = tokenizer.vocab_size
PAD = tokenizer.pad_token_id
MASK = tokenizer.mask_token_id
CLS = tokenizer.cls_token_id
BOS = tokenizer.convert_tokens_to_ids('[unused0]')
EOS = tokenizer.convert_tokens_to_ids('[unused1]')
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[unused0]', '[unused1]'],
})
