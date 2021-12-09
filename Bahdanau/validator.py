import torch.nn.functional as F
import torch
from torchtext.data.metrics import bleu_score
import spacy
from Bahdanau.decoding import GreedyDecoder, BeamSearchDecoder
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def evaluate(model, val_iterator,
             pad_idx, bos_idx, eos_idx,
             trgField=None, BLEU=False, file_dir=None):
    model.eval()
    loss = 0
    candidate = []
    reference = []
    with torch.no_grad():
        greedy = GreedyDecoder(bos_idx)
        for batch in val_iterator:
            src, src_len = batch.src
            trg, trg_len = batch.trg
            src_len = src_len.to("cpu")
            trg_len = trg_len.to("cpu")
            mask = (src != pad_idx).unsqueeze(1)  # [batch_size, 1, batch_src_max_length]
            greedy.set_params((trg.shape[1] - 1), mask)
            outputs, _ = model(src, src_len, greedy)
            loss += F.cross_entropy(outputs.reshape(-1, model.decoder.trg_vocab_size),
                                    trg[:, 1:].reshape(-1), ignore_index=pad_idx)
            if BLEU:
                decode = outputs.argmax(2)
                for i, sentence in enumerate(decode):

                    reference.append([[trgField.vocab.itos[i] for i in trg[i][1:trg_len[i] - 1].tolist()]])
                    if (sentence == eos_idx).sum() != 0:
                        length = (sentence == eos_idx).nonzero()[0] + 1
                        candidate.append([trgField.vocab.itos[i] for i in sentence[:length - 1].tolist()])
                    else:

                        candidate.append([trgField.vocab.itos[i] for i in sentence.tolist()])
        if BLEU:
            if file_dir is not None:
                with open(file_dir, 'w') as f:
                    for sentence in candidate:
                        f.write(' '.join(sentence) + '\n')
            BLEU_score = bleu_score(candidate, reference) * 100
            return loss.item() / len(val_iterator), BLEU_score
        else:
            return loss.item() / len(val_iterator)


def single_decode(model, src, dataset, decode_method, beam_width=0):
    """
    single sentence test for English to German translation
    """
    model.eval()
    with torch.no_grad():
        nlp = spacy.load('en_core_web_sm')
        src_tokens = [token.text.lower() for token in nlp.tokenizer(src.strip())]
        index = dataset.srcField.vocab.lookup_indices(src_tokens)
        bos_idx = dataset.srcField.vocab.__getitem__(dataset.special_tokens['bos'])
        eos_idx = dataset.srcField.vocab.__getitem__(dataset.special_tokens['eos'])
        index = [bos_idx] + index + [eos_idx]
        src_len = torch.tensor(len(index)).long().reshape(-1)  # [1]
        src = torch.tensor(index).to(dataset.device).unsqueeze(0)  # [1,src_len]
        if decode_method == "beam":
            beam_decoder = BeamSearchDecoder(beam_width, 50, bos_idx, eos_idx)
            sentences_idx, atten_weights = model(src, src_len, beam_decoder)
            translation = []
            for sentence_idx in sentences_idx:
                translation.append([dataset.trgField.vocab.itos[i] for i in sentence_idx])
        if decode_method == "greedy":
            greedy = GreedyDecoder(bos_idx, eos_idx, use_stop=True)
            mask = src.new_ones(1, 1, src_len)
            greedy.set_params(50, mask)
            sentence_idx, atten_weights = model(src, src_len, greedy)
            translation = [dataset.trgField.vocab.itos[i] for i in sentence_idx]
    return src_tokens, translation, atten_weights


# duplicate from https://github.com/sicnu-long/pytorch-seq2seq
def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)

    x_ticks = [''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = [''] + translation

    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
