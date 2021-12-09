# document: https://github.com/pytorch/text/tree/release/0.10/torchtext/legacy

import spacy
from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator
import io


class TranslationDataset(data.Dataset):
    """Class for using file as a datasource"""

    def __init__(self, src_path, trg_path, fields, filter_pred=None):
        """
        Create a dataset from a file and Fields
        Arguments:
            src_path : path of source sentences file
            trg_path : path of target sentences file
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist((src_line, trg_line), fields))
        self.examples = examples
        self.fields = dict(fields)
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class Dataset:
    def __init__(self, reverse_src, device, special_tokens=None):
        if special_tokens is None:
            special_tokens = {'pad': '<pad>',
                              'unk': '<unk>',
                              'bos': '<bos>',
                              'eos': '<eos>'}
        self.special_tokens = special_tokens
        self.reverse_src = reverse_src
        self.device = device
        self.train_iterator = None
        self.val_iterator = None
        self.test_iter = None

        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')
        self.srcField = Field(tokenize=self.tokenize_en,
                              init_token=self.special_tokens['bos'],
                              eos_token=self.special_tokens['eos'],
                              pad_token=self.special_tokens['pad'],
                              unk_token=self.special_tokens['unk'],
                              lower=True,
                              include_lengths=True,
                              batch_first=True)
        self.trgField = Field(tokenize=self.tokenize_de,
                              init_token=self.special_tokens['bos'],
                              eos_token=self.special_tokens['eos'],
                              pad_token=self.special_tokens['pad'],
                              unk_token=self.special_tokens['unk'],
                              lower=True,
                              include_lengths=True,
                              batch_first=True)
        fields = [('src', self.srcField), ('trg', self.trgField)]
        self.train_set = TranslationDataset('./data/multi30k/train.en', './data/multi30k/train.de', fields)
        self.val_set = TranslationDataset('./data/multi30k/val.en', './data/multi30k/val.de', fields)
        self.test_set = TranslationDataset('./data/multi30k/test2016.en', './data/multi30k/test2016.de', fields)

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        if self.reverse_src:
            return [tok.text for tok in self.spacy_en.tokenizer(text)][::-1]
        else:
            return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, src_word_minq, trg_word_minq):
        self.srcField.build_vocab(self.train_set.src, min_freq=src_word_minq)
        self.trgField.build_vocab(self.train_set.trg, min_freq=trg_word_minq)

    def build_DataLoad(self, batch_size):
        test_iter = data.Iterator(self.test_set,
                                  batch_size=batch_size,
                                  device=self.device,
                                  sort=False,
                                  sort_within_batch=False,
                                  shuffle=False,
                                  repeat=False)
        train_iterator, val_iterator = data.BucketIterator.splits(
            (self.train_set, self.val_set),
            batch_size=batch_size,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            # sort_key=lambda ex: data.interleave_keys(len(ex.src), len(ex.trg)),
            device=self.device)
        return train_iterator, val_iterator, test_iter
