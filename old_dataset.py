# unfinished

import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


def file2tokens(file_dir, language='en', lowercase=False):
    """
    Read text from file and tokenize the sentences
    """
    assert language in ('en', 'de'), \
        "tokenizer just support 'en' or 'de'"

    if language == 'en':
        nlp = spacy.load('en_core_web_md')
    if language == 'de':
        nlp = spacy.load('de_core_news_md')

    with open(file_dir) as f:
        raw_data = f.readlines()

    # create a list of sequence of tokens
    sentences = []
    for sentence in raw_data:
        if lowercase:
            sentence = [token.text.lower() for token in nlp.tokenizer(sentence.strip())]
        else:
            sentence = [token.text for token in nlp.tokenizer(sentence.strip())]

        if sentence:
            sentences.append(sentence)

    return sentences


class Data:
    def __init__(self, file_dir, lang):
        self.sentences = file2tokens(file_dir, lang, lowercase=True)
        self.vocab = None

    def build_vocab(self, min_freq, special_tokens):
        specials = [value for key, value in special_tokens.items()]
        vocab = build_vocab_from_iterator(self.sentences, min_freq, specials)
        # set the index for OOV(out of vocabulary)
        vocab.set_default_index(vocab.__getitem__(special_tokens['unk']))
        self.vocab = vocab

    def set_vocab(self, vocab):
        self.vocab = vocab

    def token2index(self, special_tokens, reverse=False):
        indexes = []
        for sentence in self.sentences:
            index = self.vocab.lookup_indices(sentence)
            if reverse:
                index = index[::-1]
            if 'bos' in special_tokens.keys():
                index = [self.vocab.__getitem__(special_tokens['bos'])] + index
            if 'eos' in special_tokens.keys():
                index = index + [self.vocab.__getitem__(special_tokens['eos'])]
            index = torch.tensor(index)
            indexes.append(index)
        return indexes


class Dataset:
    def __init__(self, src_dir, src_lang, trg_dir, trg_lang,
                 device, special_tokens=None):
        if special_tokens is None:
            special_tokens = {'pad': '<pad>',
                              'unk': '<unk>',
                              'bos': '<bos>',
                              'eos': '<eos>'}
        self.special_tokens = special_tokens
        self.src = Data(src_dir, src_lang)
        self.trg = Data(trg_dir, trg_lang)
        self.device = device

    def build_vocab(self, src_word_minq, trg_word_minq):
        self.src.build_vocab(src_word_minq, self.special_tokens)
        self.trg.build_vocab(trg_word_minq, self.special_tokens)

    def set_vocab(self, src_vocab, trg_vocab):
        self.src.set_vocab(src_vocab)
        self.trg.set_vocab(trg_vocab)

    def build_DataLoad(self, batch_size, reverse_src=False):
        src_indexes = self.src.token2index(self.special_tokens, reverse_src)
        trg_indexes = self.trg.token2index(self.special_tokens)
        pairs = list(zip(src_indexes, trg_indexes))
        return DataLoader(pairs, batch_size, shuffle=True, collate_fn=self.pad_sentence)

    def pad_sentence(self, data_pairs):
        data_pairs = sorted(data_pairs, key=lambda pairs: len(pairs[0]), reverse=True)
        src, trg = zip(*data_pairs)
        src_lens = [len(s) for s in src]
        trg_lens = [len(s) for s in trg]

        src = pad_sequence(src,
                           padding_value=self.src.vocab.__getitem__(self.special_tokens['pad']),
                           batch_first=True).to(self.device)
        trg = pad_sequence(trg,
                           padding_value=self.trg.vocab.__getitem__(self.special_tokens['pad']),
                           batch_first=True).to(self.device)
        return (src, src_lens), (trg, trg_lens)
