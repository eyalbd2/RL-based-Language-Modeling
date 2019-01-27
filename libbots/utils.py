

import string
from nltk.translate import bleu_score
from nltk.tokenize import TweetTokenizer
import numpy as np
from . import model
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_bleu_many(cand_seq, ref_sequences):
    sf = bleu_score.SmoothingFunction()
    return bleu_score.sentence_bleu(ref_sequences, cand_seq,
                                    smoothing_function=sf.method1,
                                    weights=(0.5, 0.5))

def calc_preplexity_many(probs, actions):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(torch.tensor(probs).to(device), torch.tensor(actions).to(device))
    return np.exp(loss.item())

def calc_bleu(cand_seq, ref_seq):
    return calc_bleu_many(cand_seq, [ref_seq])

def tokenize(s):
    return TweetTokenizer(preserve_case=False).tokenize(s)

def untokenize(words):
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in words]).strip()

def calc_mutual(net, back_net, p1, p2):
    # Pack forward and backward
    criterion = nn.CrossEntropyLoss()
    p2 = [1] + p2
    input_seq_forward = model.pack_input(p1, net.emb, device)
    output_seq_forward = model.pack_input(p2[:-1], net.emb, device)
    input_seq_backward = model.pack_input(tuple(p2), back_net.emb, device)
    output_seq_backward = model.pack_input(list(p1)[:-1], back_net.emb, device)
    # Enc forward and backward
    enc_forward = net.encode(input_seq_forward)
    enc_backward = back_net.encode(input_seq_backward)

    r_forward = net.decode_teacher(enc_forward, output_seq_forward).detach()
    r_backward = back_net.decode_teacher(enc_backward, output_seq_backward).detach()
    fw = criterion(torch.tensor(r_forward).to(device), torch.tensor(p2[1:]).to(device)).detach()
    bw = criterion(torch.tensor(r_backward).to(device), torch.tensor(p1[1:]).to(device)).detach()
    return (-1)*float(fw + bw)

def calc_cosine_many(mean_emb_pred, mean_emb_ref):
    norm_pred = mean_emb_pred / mean_emb_pred.norm(dim=1)[:, None]
    norm_ref = mean_emb_ref / mean_emb_ref.norm(dim=1)[:, None]
    return torch.mm(norm_pred, norm_ref.transpose(1, 0))

