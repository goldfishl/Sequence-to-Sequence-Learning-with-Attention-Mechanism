import torch


class TeacherDecoder:
    def __init__(self, teacher_forcing_ratio=0.5):
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.targets = None
        self.max_len = 0  # max decoder step
        self.mask = None

    def set_params(self, targets, mask):
        self.targets = targets
        self.max_len = targets.shape[1] - 1
        self.mask = mask

    def __call__(self, decoder, encoder_outputs, hidden_state):
        trg_vocab_size = decoder.trg_vocab_size
        batch_size, src_len, _ = encoder_outputs.shape
        outputs = encoder_outputs.new_empty(batch_size, self.max_len, trg_vocab_size)

        input_token = self.targets[:, 0]  # <eos> token [batch_size,1]
        # device, require_grad same as the  encoder_outputs
        attn_hidden = encoder_outputs.new_zeros(batch_size, 1,
                                                decoder.attn_hidden_dim)

        for i in range(self.max_len):
            out, attn_hidden, hidden_state, atten_weights = decoder(
                input_token, attn_hidden, hidden_state,
                encoder_outputs, self.mask)
            outputs[:, i, :] = out

            if torch.rand(1) < self.teacher_forcing_ratio:
                input_token = self.targets[:, 1 + i]
            else:
                input_token = out.argmax(1)
        return outputs, atten_weights


class GreedyDecoder:
    def __init__(self, bos_idx, eos_idx=None, use_stop=False):
        self.max_len = 0  # max decoder step
        self.mask = None
        self.use_stop = use_stop
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def set_params(self, max_len, mask):
        self.max_len = max_len
        self.mask = mask

    def __call__(self, decoder, encoder_outputs, hidden_state):
        trg_vocab_size = decoder.trg_vocab_size
        batch_size, src_len, _ = encoder_outputs.shape
        outputs = encoder_outputs.new_empty(batch_size, self.max_len, trg_vocab_size)
        # device, require_grad may cause problem
        input_token = encoder_outputs.new_full([batch_size], self.bos_idx).long()  # <eos> token [batch_size,1]
        attn_hidden = encoder_outputs.new_zeros(batch_size, 1, decoder.attn_hidden_dim)
        atten_weights = torch.zeros(batch_size, self.max_len, src_len)
        if self.use_stop:
            sentence_idx = []
            for i in range(self.max_len):
                out, attn_hidden, hidden_state, atten_weight = decoder(input_token, attn_hidden, hidden_state,
                                                                       encoder_outputs, self.mask)
                atten_weights[:, i, :] = atten_weight
                input_token = out.argmax(1)
                sentence_idx.append(input_token)
                if input_token == self.eos_idx:
                    atten_weights = atten_weights.squeeze(0)[:len(sentence_idx)]
                    return sentence_idx, atten_weights

        for i in range(self.max_len):
            out, attn_hidden, hidden_state, atten_weight = decoder(input_token, attn_hidden, hidden_state,
                                                                   encoder_outputs, self.mask)
            atten_weights[:, i, :] = atten_weight
            input_token = out.argmax(1)
            outputs[:, i, :] = out

        return outputs, atten_weights


class BeamSearchNode:
    def __init__(self, idx, logp, prev_node, attn_hidden, hidden_state, atten_weights):
        self.idx = idx
        self.attn_hidden = attn_hidden
        self.hidden_state = hidden_state
        self.prev_node = prev_node
        self.logp = logp
        self.atten_weights = atten_weights
        if prev_node is None:
            self.sentence_logp = logp
            self.length = 1
        else:
            self.sentence_logp = prev_node.sentence_logp + logp
            self.length = prev_node.length + 1
        self.sentence_metric = self.sentence_logp / self.length  # length normalization


class BeamSearchDecoder:
    def __init__(self, width, max_len, bos_idx, eos_idx):
        self.width = width
        self.max_len = max_len  # max decoder step
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(self, decoder, encoder_outputs, hidden_state):
        batch_size, src_len, _ = encoder_outputs.shape
        assert batch_size == 1, "Beam Search not support batch decode now"
        end_nodes = []
        best_nodes = []
        dec_step = 0
        input_token = encoder_outputs.new_full([1], self.bos_idx).long()  # <eos> token [batch_size,1]
        attn_hidden = encoder_outputs.new_zeros(1, 1, decoder.attn_hidden_dim)
        mask = encoder_outputs.new_ones(1, 1, src_len)
        out, attn_hidden, hidden_state, atten_weights = decoder(input_token, attn_hidden, hidden_state,
                                                                encoder_outputs, mask)
        out = out.squeeze().log_softmax(0)
        topk_logp, topk_idx = torch.topk(out, self.width)
        for idx, logp in zip(topk_idx, topk_logp):
            best_nodes.append(BeamSearchNode(idx, logp, None, attn_hidden, hidden_state, atten_weights))
        while len(end_nodes) < self.width:
            temp_nodes = []
            dec_step += 1
            for i, node in enumerate(best_nodes):
                out, attn_hidden, hidden_state, atten_weights = decoder(node.idx.reshape(-1),
                                                                        node.attn_hidden, node.hidden_state,
                                                                        encoder_outputs, mask)
                out = out.squeeze().log_softmax(0)
                topk_logp, topk_idx = torch.topk(out, self.width - len(end_nodes))
                for idx, logp in zip(topk_idx, topk_logp):
                    temp_nodes.append(BeamSearchNode(idx, logp, node, attn_hidden, hidden_state, atten_weights))

            temp_nodes = sorted(temp_nodes, key=lambda x: x.sentence_metric, reverse=True)[
                         :(self.width - len(end_nodes))]
            if dec_step == self.max_len:
                end_nodes += temp_nodes
            else:
                best_nodes.clear()
                for node in temp_nodes:
                    if node.idx == self.eos_idx:
                        end_nodes.append(node)
                    else:
                        best_nodes.append(node)

        sentences_idx = []
        atten_weights = []
        for node in end_nodes:
            temp_idx = []
            # temp_node = node
            atten_weight = torch.zeros(node.length, src_len)
            while node is not None:
                temp_idx.append(node.idx)
                atten_weight[node.length - 1] = node.atten_weights
                node = node.prev_node

            sentences_idx.append(temp_idx[::-1])
            atten_weights.append(atten_weight)

        return sentences_idx, atten_weights
