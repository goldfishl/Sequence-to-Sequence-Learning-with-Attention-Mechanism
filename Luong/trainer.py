import time
import torch
import torch.nn.functional as F
from Luong.decoding import TeacherDecoder
from Luong.validator import evaluate


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, optimizer, clip, dataset,
          num_epochs, mini_batch_size, accum_iter,
          print_every, teacher_forcing_ratio=0.5):
    Separator = "-----------------------------------------------------------------"
    print("batch_size:", accum_iter * mini_batch_size)
    teacher = TeacherDecoder(teacher_forcing_ratio)

    train_iterator, val_iterator, test_iter = dataset.build_DataLoad(mini_batch_size)
    pad_idx = dataset.srcField.vocab.__getitem__(dataset.special_tokens['pad'])
    bos_idx = dataset.srcField.vocab.__getitem__(dataset.special_tokens['bos'])
    eos_idx = dataset.srcField.vocab.__getitem__(dataset.special_tokens['eos'])
    batch_loss = 0
    model.train()
    for e in range(num_epochs):
        start_time = time.time()
        print(Separator + f"\nEpoch {e + 1}\n" + Separator)
        batch_idx = 0
        for mini_batch_idx, batch in enumerate(train_iterator):
            src, src_len = batch.src
            trg, trg_len = batch.trg
            src_len = src_len.to("cpu")
            trg_len = trg_len.to("cpu")
            mask = (src != pad_idx).unsqueeze(1)
            teacher.set_params(trg, mask)
            outputs, atten_weights = model(src, src_len, teacher)  # bug
            mini_batch_loss = F.cross_entropy(
                outputs.reshape(-1, model.decoder.trg_vocab_size),
                trg[:, 1:].reshape(-1),
                ignore_index=pad_idx)
            mini_batch_loss /= accum_iter
            mini_batch_loss.backward()
            batch_loss += mini_batch_loss.item()
            # gradient accumulate
            if (mini_batch_idx + 1) % accum_iter == 0:
                batch_idx += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad()
                # print training information
                if (batch_idx % print_every) == 0 or (mini_batch_idx + 1) == len(train_iterator):
                    val_loss, BLEU_score = evaluate(model, val_iterator,
                                                    pad_idx, bos_idx, eos_idx,
                                                    trgField=dataset.trgField, BLEU=True)
                    print("[ {}/{} ]".format(mini_batch_idx + 1, len(train_iterator)),
                          "Loss: {:.6f}...".format(batch_loss),
                          "Val Loss: {:.6f}...".format(val_loss),
                          "Val BLEU: {:.6f}...".format(BLEU_score))
                    if (mini_batch_idx + 1) == len(train_iterator):
                        end_time = time.time()
                        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                        epoch_loss, BLEU_score = evaluate(model, train_iterator,
                                                          pad_idx, bos_idx, eos_idx,
                                                          trgField=dataset.trgField, BLEU=True)
                        print(f"Total Epoche:  |    Time: {epoch_mins}m {epoch_secs}s\n",
                              "Epoch Loss: {:.6f}...".format(epoch_loss),
                              "Train BLEU: {:.6f}...".format(BLEU_score))
                batch_loss = 0
                model.train()  # because model.eval() will be called in evaluate
