

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from . import utils, data
from torch.autograd import Variable


HIDDEN_STATE_SIZE = 512
EMBEDDING_DIM = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True)
        self.output = nn.Sequential(nn.Linear(hid_size, dict_size))

    def encode(self, x):
        _, hid = self.encoder(x)
        return hid

    def get_encoded_item(self, encoded, index):
        # For RNN
        # return encoded[:, index:index+1]
        # For LSTM
        return encoded[0][:, index:index+1].contiguous(), \
               encoded[1][:, index:index+1].contiguous()

    def decode_teacher(self, hid, input_seq, detach=False):
        # Method assumes batch of size=1
        if detach:
            out, _ = self.decoder(input_seq, hid)
            out = self.output(out.data).detach()
        else:
            out, _ = self.decoder(input_seq, hid)
            out = self.output(out.data)
        return out

    def decode_one(self, hid, input_x, detach=False):
        if detach:
            out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
            out = self.output(out).detach()
        else:
            out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
            out = self.output(out)
        return out.squeeze(dim=0), new_hid

    def decode_chain_argmax(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        """
        res_logits = []
        res_tokens = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_token_v = torch.max(out_logits, dim=1)[1]
            out_token = out_token_v.data.cpu().numpy()[0]

            cur_emb = self.emb(out_token_v)

            res_logits.append(out_logits)
            res_tokens.append(out_token)
            if stop_at_token is not None and out_token == stop_at_token:
                break
        return torch.cat(res_logits), res_tokens

    def decode_chain_sampling(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        """
        res_logits = []
        res_actions = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.data.cpu().numpy()[0]
            action = np.random.choice(out_probs.shape[0], p=out_probs)
            action_v = torch.LongTensor([action]).to(begin_emb.device)
            cur_emb = self.emb(action_v)

            res_logits.append(out_logits)
            res_actions.append(action)
            if stop_at_token is not None and action == stop_at_token:
                break
        return torch.cat(res_logits), res_actions

    def decode_rl_chain_argmax(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        """
        q_list = []
        res_tokens = []
        hidden_list = []
        emb_list = []
        cur_emb = begin_emb
        hidden_list.append(hid)
        emb_list.append(cur_emb)


        for _ in range(seq_len):
            out_q, hid = self.decode_one(hid, cur_emb)
            out_token_v = torch.max(out_q, dim=1)[1]
            out_token = out_token_v.data.cpu().numpy()[0]

            cur_emb = self.emb(out_token_v)


            hidden_list.append(hid)
            emb_list.append(cur_emb)
            q_list.append(out_q[0][out_token])
            res_tokens.append(out_token)
            if stop_at_token is not None and out_token == stop_at_token:
                break
        return q_list, hidden_list, torch.cat(emb_list), res_tokens

    def decode_rl_chain_sampling(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        """
        q_list = []
        res_tokens = []
        hidden_list = []
        emb_list = []

        cur_emb = begin_emb
        hidden_list.append(hid)
        emb_list.append(cur_emb)

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.data.cpu().numpy()[0]
            action = int(np.random.choice(out_probs.shape[0], p=out_probs))
            action_v = torch.LongTensor([action]).to(begin_emb.device)
            cur_emb = self.emb(action_v)

            q_list.append(out_logits[0][action])
            res_tokens.append(action)
            hidden_list.append(hid)
            emb_list.append(cur_emb)
            if stop_at_token is not None and action == stop_at_token:
                break
        return q_list, hidden_list, torch.cat(emb_list), res_tokens

    def decode_batch(self, state):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        """
        res_logits = []
        for idx in range(len(state)):
            cur_q, _ = self.decode_one(state[idx][0], state[idx][1].unsqueeze(0))
            res_logits.append(cur_q)

        return torch.cat(res_logits)

    def decode_k_best(self, hid, begin_emb, k, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        """
        cur_emb = begin_emb

        out_logits, hid = self.decode_one(hid, cur_emb)
        out_probs_v = F.log_softmax(out_logits, dim=1)
        highest_values = torch.topk(out_probs_v, k)
        first_beam_probs = highest_values[0]
        first_beam_vals = highest_values[1]
        beam_emb = []
        beam_hid = []
        beam_vals = []
        beam_prob = []
        for i in range(k):
            beam_vals.append([])
        for i in range(k):
            beam_emb.append(self.emb(first_beam_vals)[:, i, :])
            beam_hid.append(hid)
            beam_vals[i].append(int(first_beam_vals[0][i].data.cpu().numpy()))
            beam_prob.append(first_beam_probs[0][i].data.cpu().numpy())

        for num_words in range(1, seq_len):
            possible_sentences = []
            possible_probs = []
            for i in range(k):
                cur_emb = beam_emb[i]
                cur_hid = beam_hid[i]
                cur_prob = beam_prob[i]
                if stop_at_token is not None and beam_vals[i][-1] == stop_at_token:
                    possible_probs.append(cur_prob)
                    possible_sentences.append((beam_vals[i], beam_hid[i], beam_emb[i]))
                    continue

                else:
                    out_logits, hid = self.decode_one(cur_hid, cur_emb)
                    log_probs = torch.tensor(F.log_softmax(out_logits, dim=1)).to(device)
                    highest_values = torch.topk(log_probs, k)
                    probs = highest_values[0]
                    vals = highest_values[1]
                    for j in range(k):
                        prob = (probs[0][j].data.cpu().numpy() + cur_prob) * (num_words / (num_words + 1))
                        possible_probs.append(prob)
                        emb = self.emb(vals)[:, j, :]
                        val = vals[0][j].data.cpu().numpy()
                        cur_seq_vals = beam_vals[i][:]
                        cur_seq_vals.append(int(val))
                        possible_sentences.append((cur_seq_vals, hid, emb))

            max_indices = sorted(range(len(possible_probs)), key=lambda s: possible_probs[s])[-k:]
            beam_emb = []
            beam_hid = []
            beam_vals = []
            beam_prob = []
            for i in range(k):
                (seq_vals, hid, emb) = possible_sentences[max_indices[i]]
                beam_emb.append(emb)
                beam_hid.append(hid)
                beam_vals.append(seq_vals)
                beam_prob.append(possible_probs[max_indices[i]])
        t_beam_prob = []
        for i in range(k):
            t_beam_prob.append(Variable(torch.tensor(beam_prob[i]).to(device), requires_grad=True))

        return beam_vals, t_beam_prob

    def decode_k_sampling(self, hid, begin_emb, k, seq_len, stop_at_token=None):
        list_res_logit = []
        list_res_actions = []
        for i in range(k):
            res_logits = []
            res_actions = []
            cur_emb = begin_emb

            for _ in range(seq_len):
                out_logits, hid = self.decode_one(hid, cur_emb)
                out_probs_v = F.softmax(out_logits, dim=1)
                out_probs = out_probs_v.data.cpu().numpy()[0]
                action = int(np.random.choice(out_probs.shape[0], p=out_probs))
                action_v = torch.LongTensor([action]).to(begin_emb.device)
                cur_emb = self.emb(action_v)

                res_logits.append(out_logits)
                res_actions.append(action)
                if stop_at_token is not None and action == stop_at_token:
                    break

            list_res_logit.append(torch.cat(res_logits))
            list_res_actions.append(res_actions)

        return list_res_actions, list_res_logit

    def get_qp_prob(self, hid, begin_emb, p_tokens):
        """
        Find the probability of P(q|p) - the question asked (q, source) given the answer (p, target)
        """
        cur_emb = begin_emb
        prob = 0
        for word_num in range(len(p_tokens)):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_token_v = F.log_softmax(out_logits, dim=1)[0]
            prob = prob + out_token_v[p_tokens[word_num]].data.cpu().numpy()

            out_token = p_tokens[word_num]
            action_v = torch.LongTensor([out_token]).to(begin_emb.device)
            cur_emb = self.emb(action_v)
        if (len(p_tokens)) == 0:
            return -20
        else:
            return prob/(len(p_tokens))

    def get_mean_emb(self, begin_emb, p_tokens):
        """
        Find the embbeding of the entire sentence
        """
        emb_list = []
        cur_emb = begin_emb
        emb_list.append(cur_emb)
        if isinstance(p_tokens, list):
            if len(p_tokens) == 1:
                p_tokens = p_tokens[0]
            for word_num in range(len(p_tokens)):
                out_token = p_tokens[word_num]
                action_v = torch.LongTensor([out_token]).to(begin_emb.device)
                cur_emb = self.emb(action_v).detach()
                emb_list.append(cur_emb)
        else:
            out_token = p_tokens
            action_v = torch.LongTensor([out_token]).to(begin_emb.device)
            cur_emb = self.emb(action_v).detach()
            emb_list.append(cur_emb)

        return sum(emb_list)/len(emb_list)

    def get_beam_sentences(self, tokens, emb_dict, k_sentences):
        # Forward
        source_seq = pack_input(tokens, self.emb)
        enc = self.encode(source_seq)
        end_token = emb_dict[data.END_TOKEN]
        probs, list_of_out_tokens = self.decode_k_sampling(enc, source_seq.data[0:1], k_sentences,
                                                               seq_len=data.MAX_TOKENS, stop_at_token=end_token)
        # list_of_out_tokens, probs = self.decode_k_sent(enc, source_seq.data[0:1], k_sentences, seq_len=data.MAX_TOKENS,
        #                                               stop_at_token=end_token)
        return list_of_out_tokens, probs

    def get_action_prob(self, source, target, emb_dict):
        source_seq = pack_input(source, self.emb, "cuda")
        hid = self.encode(source_seq)
        end_token = emb_dict[data.END_TOKEN]
        cur_emb = source_seq.data[0:1]
        probs = []
        for word_num in range(len(target)):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_token_v = F.log_softmax(out_logits, dim=1)[0]

            probs.append(out_token_v[target[word_num]])
            out_token = target[word_num]
            action_v = torch.LongTensor([out_token]).to(device)
            cur_emb = self.emb(action_v)
        t_prob = probs[0]
        for i in range(1, len(probs)):
            t_prob = t_prob + probs[i]

        return t_prob/len(probs)

    def get_logits(self, hid, begin_emb, seq_len, res_action, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        """
        res_logits = []
        cur_emb = begin_emb

        for i in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb, True)
            out_token_v = torch.tensor([res_action[i]]).to(device)
            out_token = out_token_v.data.cpu().numpy()[0]

            cur_emb = self.emb(out_token_v).detach()

            res_logits.append(out_logits)
            if stop_at_token is not None and out_token == stop_at_token:
                break
        return torch.cat(res_logits)



def pack_batch_no_out(batch, embeddings, device=device):
    assert isinstance(batch, list)
    # Sort descending (CuDNN requirements)
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    input_idx, output_idx = zip(*batch)
    # create padded matrix of inputs
    lens = list(map(len, input_idx))
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x
    input_v = torch.tensor(input_mat).to(device)
    input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
    # lookup embeddings
    r = embeddings(input_seq.data)
    emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)
    return emb_input_seq, input_idx, output_idx

def pack_batch_no_in(batch, embeddings, device=device):
    assert isinstance(batch, list)
    # Sort descending (CuDNN requirements)
    batch.sort(key=lambda s: len(s[1]), reverse=True)
    input_idx, output_idx = zip(*batch)
    # create padded matrix of inputs
    lens = list(map(len, output_idx))
    output_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    for idx, x in enumerate(output_idx):
        output_mat[idx, :len(x)] = x
    ouput_v = torch.tensor(output_mat).to(device)
    output_seq = rnn_utils.pack_padded_sequence(ouput_v, lens, batch_first=True)
    # lookup embeddings
    r = embeddings(output_seq.data)
    emb_input_seq = rnn_utils.PackedSequence(r, output_seq.batch_sizes)
    return emb_input_seq, output_idx, input_idx

def pack_input(input_data, embeddings, device=device, detach=False):
    input_v = torch.LongTensor([input_data]).to(device)
    if detach:
        r = embeddings(input_v).detach()
    else:
        r = embeddings(input_v)
    return rnn_utils.pack_padded_sequence(r, [len(input_data)], batch_first=True)

def pack_batch(batch, embeddings, device=device, detach=False):
    emb_input_seq, input_idx, output_idx = pack_batch_no_out(batch, embeddings, device)

    # prepare output sequences, with end token stripped
    output_seq_list = []
    for out in output_idx:
        if len(out) == 1:
            out = out[0]
        output_seq_list.append(pack_input(out[:-1], embeddings, device, detach))
    return emb_input_seq, output_seq_list, input_idx, output_idx

def pack_backward_batch(batch, embeddings, device=device):
    emb_target_seq, target_idx, source_idx = pack_batch_no_in(batch, embeddings, device)

    # prepare output sequences, with end token stripped
    source_seq = []
    for src in source_idx:
        source_seq.append(pack_input(src[:-1], embeddings, device))
    backward_input = emb_target_seq
    backward_output = source_seq
    return backward_input, backward_output, target_idx, source_idx

def seq_bleu(model_out, ref_seq):
    model_seq = torch.max(model_out.data, dim=1)[1]
    model_seq = model_seq.cpu().numpy()
    return utils.calc_bleu(model_seq, ref_seq)

def mutual_words_to_words(words, data, k,  emb_dict, rev_emb_dict, net, back_net):
    # Forward
    tokens = data.encode_words(words, emb_dict)
    source_seq = pack_input(tokens, net.emb, "cuda")
    enc = net.encode(source_seq)
    end_token = emb_dict[data.END_TOKEN]
    list_of_out_tokens, probs = net.decode_k_best(enc, source_seq.data[0:1], k, seq_len=data.MAX_TOKENS,
                                           stop_at_token=end_token)
    list_of_out_words = []
    for iTokens in range(len(list_of_out_tokens)):
        if list_of_out_tokens[iTokens][-1] == end_token:
            list_of_out_tokens[iTokens] = list_of_out_tokens[iTokens][:-1]
        list_of_out_words.append(data.decode_words(list_of_out_tokens[iTokens], rev_emb_dict))

    # Backward
    back_seq2seq_prob = []
    for iTarget in range(len(list_of_out_words)):
        b_tokens = data.encode_words(list_of_out_words[iTarget], emb_dict)
        target_seq = pack_input(b_tokens, back_net.emb, "cuda")
        b_enc = back_net.encode(target_seq)
        back_seq2seq_prob.append(back_net.get_qp_prob(b_enc, target_seq.data[0:1], tokens[1:]))

    mutual_prob = []
    for i in range(len(probs)):
        mutual_prob.append(probs[i] + back_seq2seq_prob[i])

    return list_of_out_words, mutual_prob






