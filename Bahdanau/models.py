from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from Bahdanau.hyperparameter import hp


class Encoder(nn.Module):
    def __init__(self, src_pad_idx, src_vocab_size, embedding_dim,
                 enc_hidden_dim, dec_hidden_dim, num_layers=1, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=src_pad_idx)
        self.rnn = nn.GRU(embedding_dim, enc_hidden_dim, num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        # embedded = self.embedding(src)
        packed_embedded = pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)

        packed_outputs, hidden_state = self.rnn(packed_embedded)  # packed_outputs: PackSequence

        enc_outputs, _ = pad_packed_sequence(packed_outputs,
                                             batch_first=True)  # [batch_size, batch_src_max_length, hidden_dim]

        # hidden [-2, :, : ] is the last of the forwards RNN [batch_size,enc_hidden_dim]
        # hidden [-1, :, : ] is the last of the backwards RNN [batch_size,enc_hidden_dim]
        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        # hidden_state:[batch_size,dec_hidden_dim]
        hidden_state = torch.tanh(self.fc(torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)))
        return enc_outputs, hidden_state.unsqueeze(0)


class Bahdanau_Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.W = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, encoder_outputs, mask):
        batch, src_len, _ = encoder_outputs.shape  # [batch_size, batch_src_max_length, enc_hidden_dim * 2]

        dec_hidden = dec_hidden.permute(1, 0, 2)  # [1,batch_size,dec_hidden_dim] to [batch_size,1,dec_hidden_dim]
        dec_hidden = dec_hidden.repeat(1, src_len, 1)  # [batch_size,src_len,enc_hidden_dim]
        energy = self.W(torch.cat((dec_hidden, encoder_outputs), dim=2)).tanh()
        atten_score = self.v(energy).permute(0, 2, 1)  # [batch size, 1, batch_src_max_length]

        # mask:[1,batch_src_max_length] atten_score:[batch_size,1,batch_src_max_length]
        atten_score = atten_score.masked_fill(mask == 0, -1e10)
        atten_weights = atten_score.softmax(2)
        context = torch.bmm(atten_weights,  # [batch_size,1,batch_src_max_length]
                            encoder_outputs  # [batch_size,batch_src_max_length,enc_hidden_dim * 2]
                            )  # [batch_size,1,enc_hidden_dim * 2]

        return context, atten_weights.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, trg_pad_idx, trg_vocab_size, embedding_dim,
                 enc_hidden_dim, dec_hidden_dim, num_layers, dropout=0):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size  # for TeacherDecoding
        self.embedding = nn.Embedding(trg_vocab_size, embedding_dim, padding_idx=trg_pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_dim + (enc_hidden_dim * 2),  # feed-input tech
                          dec_hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=False)

        self.attention = Bahdanau_Attention(enc_hidden_dim, dec_hidden_dim)
        self.fc_out = nn.Linear(embedding_dim + (enc_hidden_dim * 2) + dec_hidden_dim,
                                trg_vocab_size)

    def forward(self, trg, hidden_state, encoder_outputs, mask):
        trg = trg.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(trg))
        # embedded = self.embedding(trg)

        context, atten_weights = self.attention(hidden_state, encoder_outputs,
                                                mask)  # context:[batch_size,1,hidden_dim]

        decoder_hidden, hidden_state = self.rnn(torch.cat([embedded, context], dim=2), hidden_state)

        # out before squeeze:[batch_size, 1, trg_vocab_size]; after squeeze:[batch_size,trg_vocab_size]
        out = self.fc_out(torch.cat((embedded, context, decoder_hidden), dim=2)).squeeze(1)

        return out, hidden_state, atten_weights


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, build=True):
        super().__init__()
        if build:
            self.params = {
                'src_vocab_size': src_vocab_size,
                'trg_vocab_size': trg_vocab_size,
                'embedding_dim': hp.embedding_dim,
                'enc_hidden_dim': hp.enc_hidden_dim,
                'dec_hidden_dim': hp.dec_hidden_dim,
                'num_layers': hp.num_layers,
                'dropout': hp.dropout,
                'src_pad_idx': src_pad_idx,
                'trg_pad_idx': trg_pad_idx
            }
            self.build()

    def build(self):
        self.encoder = Encoder(self.params['src_pad_idx'],
                               self.params['src_vocab_size'],
                               self.params['embedding_dim'],
                               self.params['enc_hidden_dim'],
                               self.params['dec_hidden_dim'],
                               self.params['num_layers'],
                               self.params['dropout'])
        self.decoder = Decoder(self.params['trg_pad_idx'],
                               self.params['trg_vocab_size'],
                               self.params['embedding_dim'],
                               self.params['enc_hidden_dim'],
                               self.params['dec_hidden_dim'],
                               self.params['num_layers'],
                               self.params['dropout'])

    def save(self, file):
        torch.save((self.params, self.state_dict()), file)

    def load(self, file, device='cpu'):
        self.params, state_dict = torch.load(file)
        self.build()
        self.load_state_dict(state_dict)
        self.to(device)

    def forward(self, src, src_len, decodingModule=None):
        encoder_outputs, hidden_state = self.encoder(src, src_len)
        return decodingModule(self.decoder, encoder_outputs, hidden_state)



