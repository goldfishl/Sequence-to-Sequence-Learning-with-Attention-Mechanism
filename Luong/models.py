from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from Luong.hyperparameter import hp


class Encoder(nn.Module):
    def __init__(self, src_pad_idx, src_vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=src_pad_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        # embedded = self.embedding(src)
        packed_embedded = pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)

        packed_outputs, hidden_state = self.rnn(packed_embedded)  # packed_outputs: PackSequence

        enc_outputs, _ = pad_packed_sequence(packed_outputs,
                                             batch_first=True)  # [batch_size, batch_src_max_length, hidden_dim]

        return enc_outputs, hidden_state


class Luong_Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, hidden_dim):
        super(Luong_Attention, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim

        if self.method == 'general':
            # self.attention = nn.Bilinear(hidden_dim,hidden_dim,1)
            # self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W = nn.Linear(hidden_dim, hidden_dim)

        elif self.method == 'concat':
            self.W = nn.Linear(self.hidden_dim * 2, 1, bias=False)
            # self.W = nn.Linear(self.hidden_dim * 2, 1)

        elif self.method == 'MLP':
            # self.W = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.W = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.v = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        batch, src_len, _ = encoder_outputs.shape
        # batch vector dot computation
        if self.method == 'dot':
            atten_score = torch.bmm(decoder_hidden,  # [batch_size,1,hidden_dim]
                                    encoder_outputs.permute(0, 2, 1)  # [batch_size,hidden_dim,batch_src_max_length]
                                    )
            # mask the pad token in sequence

            atten_score = atten_score.masked_fill(mask == 0, -1e10)  # mask:[batch_size,1,batch_src_max_length]
            atten_weights = atten_score.softmax(2)
            context = torch.bmm(atten_weights,  # [batch_size,1,batch_src_max_length]
                                encoder_outputs  # [batch_size,batch_src_max_length,hidden_dim]
                                )  # [batch_size,1,hidden_dim]

        elif self.method == 'general':
            # batch vector dot computation
            atten_score = torch.bmm(self.W(decoder_hidden),  # [batch_size,1,hidden_dim]
                                    encoder_outputs.permute(0, 2, 1)  # [batch_size,hidden_dim,batch_src_max_length]
                                    )  # [batch_size,1,batch_src_max_length]
            # mask the pad token in sequence
            atten_score = atten_score.masked_fill(mask == 0, -1e10)  # mask:[batch_size,1,batch_src_max_length]
            atten_weights = atten_score.softmax(2)
            context = torch.bmm(atten_weights,  # [batch_size,1,batch_src_max_length]
                                encoder_outputs  # [batch_size,batch_src_max_length,hidden_dim]
                                )  # [batch_size,1,hidden_dim]

        # if self.method == 'general':
        #     # batch vector dot computation
        #     atten_score = torch.bmm(self.W(encoder_outputs),  # [batch_size,batch_src_max_length,hidden_dim]
        #                             decoder_hidden.permute(0, 2, 1)  # [batch_size,hidden_dim,1]
        #                             )  # [batch_size,batch_src_max_length,1]
        #     # mask the pad token in sequence
        #     atten_score = atten_score.permute(0, 2, 1).masked_fill(mask == 0, -1e10)  # mask:[batch_size,1,batch_src_max_length]
        #     # atten_weights = atten_score.softmax(2)
        #     atten_weights = atten_score.softmax(2)
        #     context = torch.bmm(atten_weights,  # [batch_size,1,batch_src_max_length]
        #                         encoder_outputs  # [batch_size,batch_src_max_length,hidden_dim]
        #                         )  # [batch_size,1,hidden_dim]

        elif self.method == 'concat':
            decoder_hidden = decoder_hidden.repeat(1, src_len, 1)
            atten_score = self.W(torch.cat((decoder_hidden, encoder_outputs), dim=2))  # [batch_size,src_len,1]
            atten_score = atten_score.permute(0, 2, 1)  # [batch_size,1,src_len]
            atten_score = atten_score.masked_fill(mask == 0, -1e10)  # mask:[batch_size,1,batch_src_max_length]
            atten_weights = atten_score.softmax(2)
            context = torch.bmm(atten_weights,  # [batch_size,1,batch_src_max_length]
                                encoder_outputs  # [batch_size,batch_src_max_length,hidden_dim]
                                )  # [batch_size,1,hidden_dim]

        elif self.method == 'MLP':
            decoder_hidden = decoder_hidden.expand_as(encoder_outputs)
            energy = self.W(torch.cat((decoder_hidden, encoder_outputs), dim=2)).tanh()
            atten_score = self.v(energy)  # [batch_size, batch_src_max_length, 1]
            # mask the pad token in sequence
            atten_score = atten_score.permute(0, 2, 1)  # [batch_size, 1, batch_src_max_length]
            atten_score = atten_score.masked_fill(mask == 0, -1e10)  # mask:[batch_size,1,batch_src_max_length]
            atten_weights = atten_score.softmax(2)
            context = torch.bmm(atten_weights,  # [batch_size,1,batch_src_max_length]
                                encoder_outputs  # [batch_size,batch_src_max_length,hidden_dim]
                                )  # [batch_size,1,hidden_dim]

        return context, atten_weights.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, trg_pad_idx, attn_model,
                 trg_vocab_size, embedding_dim,
                 hidden_dim, attn_hidden_dim,
                 num_layers, dropout=0):
        super().__init__()
        self.trg_vocab_size = trg_vocab_size  # for TeacherDecoding
        self.attn_hidden_dim = attn_hidden_dim
        self.embedding = nn.Embedding(trg_vocab_size,
                                      embedding_dim,
                                      padding_idx=trg_pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(embedding_dim + attn_hidden_dim,  # feed-input tech
                          hidden_dim,
                          num_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=False)

        self.attention = Luong_Attention(attn_model, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, attn_hidden_dim)
        self.out_fc = nn.Linear(attn_hidden_dim, trg_vocab_size)

    def forward(self, trg, attn_hidden, hidden_state, encoder_outputs, mask):
        trg = trg.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(trg))
        # embedded = self.embedding(trg)

        decoder_hidden, hidden_state = self.rnn(torch.cat([embedded, attn_hidden], dim=2), hidden_state)
        context, atten_weights = self.attention(decoder_hidden, encoder_outputs,
                                                mask)  # context:[batch_size,1,hidden_dim]

        # attn_hidden = self.dropout(self.fc(torch.cat([decoder_hidden, context], 2))).tanh()
        attn_hidden = self.fc(torch.cat([decoder_hidden, context], 2)).tanh()
        # attn_hidden:[batch_size,1,attn_hidden_dim]
        out = self.out_fc(attn_hidden).squeeze(1)
        # before squeeze:[batch_size, 1, trg_vocab_size]; after squeeze:[batch_size,trg_vocab_size]

        return out, attn_hidden, hidden_state, atten_weights


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, build=True):
        super().__init__()
        if build:
            self.params = {
                'src_vocab_size': src_vocab_size,
                'trg_vocab_size': trg_vocab_size,
                'embedding_dim': hp.embedding_dim,
                'hidden_dim': hp.hidden_dim,
                'num_layers': hp.num_layers,
                'dropout': hp.dropout,
                'attn_model': hp.attn_model,
                'attn_hidden_dim': hp.attn_hidden_dim,
                'src_pad_idx': src_pad_idx,
                'trg_pad_idx': trg_pad_idx
            }
            self.build()

    def build(self):
        self.encoder = Encoder(self.params['src_pad_idx'],
                               self.params['src_vocab_size'],
                               self.params['embedding_dim'],
                               self.params['hidden_dim'],
                               self.params['num_layers'],
                               self.params['dropout'])
        self.decoder = Decoder(self.params['trg_pad_idx'],
                               self.params['attn_model'],
                               self.params['trg_vocab_size'],
                               self.params['embedding_dim'],
                               self.params['hidden_dim'],
                               self.params['attn_hidden_dim'],
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
