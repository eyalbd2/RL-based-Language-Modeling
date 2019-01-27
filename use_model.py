

import os
import logging
import torch

from libbots import data, model, utils


log = logging.getLogger("use")
k_sentences = 100

def words_to_words(words, emb_dict, rev_emb_dict, net, use_sampling=False):
    tokens = data.encode_words(words, emb_dict)
    input_seq = model.pack_input(tokens, net.emb)
    enc = net.encode(input_seq)
    end_token = emb_dict[data.END_TOKEN]
    if use_sampling:
        _, out_tokens = net.decode_chain_sampling(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                                  stop_at_token=end_token)
    else:
        _, out_tokens = net.decode_chain_argmax(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                                stop_at_token=end_token)
    if out_tokens[-1] == end_token:
        out_tokens = out_tokens[:-1]
    out_words = data.decode_words(out_tokens, rev_emb_dict)
    return out_words

def mutual_words_to_words(words, emb_dict, rev_emb_dict, back_emb_dict, net, back_net):
    # Forward
    tokens = data.encode_words(words, emb_dict)
    source_seq = model.pack_input(tokens, net.emb)
    enc = net.encode(source_seq)
    end_token = emb_dict[data.END_TOKEN]
    list_of_out_tokens, probs = net.decode_k_best(enc, source_seq.data[0:1], k_sentences, seq_len=data.MAX_TOKENS,
                                           stop_at_token=end_token)
    list_of_out_words = []
    for iTokens in range(len(list_of_out_tokens)):
        if list_of_out_tokens[iTokens][-1] == end_token:
            list_of_out_tokens[iTokens] = list_of_out_tokens[iTokens][:-1]
        list_of_out_words.append(data.decode_words(list_of_out_tokens[iTokens], rev_emb_dict))

    # Backward
    back_seq2seq_prob = []
    for iTarget in range(len(list_of_out_words)):
        b_tokens = data.encode_words(list_of_out_words[iTarget], back_emb_dict)
        target_seq = model.pack_input(b_tokens, back_net.emb)
        b_enc = back_net.encode(target_seq)
        back_seq2seq_prob.append(back_net.get_qp_prob(b_enc, target_seq.data[0:1], tokens[1:]))

    mutual_prob = []
    for i in range(len(probs)):
        mutual_prob.append(probs[i] + back_seq2seq_prob[i])
    most_prob_mutual_sen_id = sorted(range(len(mutual_prob)), key=lambda s: mutual_prob[s])[-1:][0]

    return list_of_out_words[most_prob_mutual_sen_id], mutual_prob[most_prob_mutual_sen_id]

def process_string(words, emb_dict, rev_emb_dict, net, use_sampling=False):
    out_words = words_to_words(words, emb_dict, rev_emb_dict, net, use_sampling=use_sampling)
    print(" ".join(out_words))


logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

load_seq2seq_path = 'Final_Saves/seq2seq/epoch_090_0.800_0.107.dat'
laod_b_seq2seq_path = 'Final_Saves/backward_seq2seq/epoch_080_0.780_0.104.dat'
bleu_model_path = 'Final_Saves/RL_BLUE/bleu_0.135_177.dat'
mutual_model_path = 'Final_Saves/RL_Mutual/epoch_180_-4.325_-7.192.dat'
prep_model_path = 'Final_Saves/RL_Perplexity/epoch_050_1.463_3.701.dat'
cos_model_path = 'Final_Saves/RL_COSINE/cosine_0.621_03.dat'

input_sentences = []
input_sentences.append("Do you want to stay?")
input_sentences.append("What's your full name?")
input_sentences.append("Where are you going?")
input_sentences.append("How old are you?")
input_sentences.append("Where are you?")
input_sentences.append("hi, joey.")
input_sentences.append("let's go.")
input_sentences.append("excuse me?")
input_sentences.append("what's that?")
input_sentences.append("Stop!")
input_sentences.append("where ya goin?")
input_sentences.append("what's this?")
input_sentences.append("Do you play football?")
input_sentences.append("who is she?")
input_sentences.append("who is he?")
input_sentences.append("Are you sure?")
input_sentences.append("Did you see that?")
input_sentences.append("Hello.")

sample = False
mutual = True
RL = True
self = 1
device = torch.device("cuda")  # "cuda"/"cpu"

# Load Seq2Seq
seq2seq_emb_dict = data.load_emb_dict(os.path.dirname(load_seq2seq_path))
seq2seq_rev_emb_dict = {idx: word for word, idx in seq2seq_emb_dict.items()}
net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(seq2seq_emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
net.load_state_dict(torch.load(load_seq2seq_path))

# Load Back Seq2Seq
b_seq2seq_emb_dict = data.load_emb_dict(os.path.dirname(laod_b_seq2seq_path))
b_seq2seq_rev_emb_dict = {idx: word for word, idx in b_seq2seq_emb_dict.items()}
b_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(b_seq2seq_emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
b_net.load_state_dict(torch.load(laod_b_seq2seq_path))

# Load BLEU
bleu_emb_dict = data.load_emb_dict(os.path.dirname(bleu_model_path))
bleu_rev_emb_dict = {idx: word for word, idx in bleu_emb_dict.items()}
bleu_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(bleu_emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
bleu_net.load_state_dict(torch.load(bleu_model_path))

# Load Mutual
mutual_emb_dict = data.load_emb_dict(os.path.dirname(mutual_model_path))
mutual_rev_emb_dict = {idx: word for word, idx in mutual_emb_dict.items()}
mutual_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(mutual_emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
mutual_net.load_state_dict(torch.load(mutual_model_path))


# Load Preplexity
prep_emb_dict = data.load_emb_dict(os.path.dirname(prep_model_path))
prep_rev_emb_dict = {idx: word for word, idx in prep_emb_dict.items()}
prep_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(prep_emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
prep_net.load_state_dict(torch.load(prep_model_path))

# Load Preplexity
prep1_emb_dict = data.load_emb_dict(os.path.dirname(prep_model_path))
prep1_rev_emb_dict = {idx: word for word, idx in prep1_emb_dict.items()}
prep1_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(prep1_emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
prep1_net.load_state_dict(torch.load(prep_model_path))

# Load Cosine Similarity
cos_emb_dict = data.load_emb_dict(os.path.dirname(cos_model_path))
cos_rev_emb_dict = {idx: word for word, idx in cos_emb_dict.items()}
cos_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(cos_emb_dict), hid_size=model.HIDDEN_STATE_SIZE).to(device)
cos_net.load_state_dict(torch.load(cos_model_path))

while True:
    input_sentence = input_sentences[0]
    if input_sentence:
        input_string = input_sentence
    else:
        input_string = input(">>> ")
    if not input_string:
        break

    words = utils.tokenize(input_string)
    for _ in range(self):
        if RL:
            words_seq2seq = words_to_words(words, seq2seq_emb_dict, seq2seq_rev_emb_dict, net, use_sampling=sample)
            words_bleu = words_to_words(words, bleu_emb_dict, bleu_rev_emb_dict, bleu_net, use_sampling=sample)
            words_mutual_RL = words_to_words(words, mutual_emb_dict, mutual_rev_emb_dict, mutual_net, use_sampling=sample)
            words_mutual, _ = mutual_words_to_words(words, seq2seq_emb_dict, seq2seq_rev_emb_dict, b_seq2seq_emb_dict,
                                             net, b_net)

            words_prep = words_to_words(words, prep_emb_dict, prep_rev_emb_dict, prep_net, use_sampling=sample)
            words_prep = words_to_words(words, prep1_emb_dict, prep1_rev_emb_dict, prep1_net, use_sampling=sample)
            words_cosine = words_to_words(words, cos_emb_dict, cos_rev_emb_dict, cos_net, use_sampling=sample)

            print('Seq2Seq: ', utils.untokenize(words_seq2seq))
            print('BLEU: ', utils.untokenize(words_bleu))
            print('Mutual Information (RL): ', utils.untokenize(words_mutual_RL))
            print('Mutual Information: ', utils.untokenize(words_mutual))
            print('Perplexity: ', utils.untokenize(words_prep))
            print('Perplexity: ', utils.untokenize(words_prep))
            print('Cosine Similarity: ', utils.untokenize(words_cosine))
        else:
            if mutual:
                words, _ = mutual_words_to_words(words, seq2seq_emb_dict, seq2seq_rev_emb_dict, b_seq2seq_emb_dict,
                                                 net, b_net)
            else:
                words = words_to_words(words, seq2seq_emb_dict, seq2seq_rev_emb_dict, net, use_sampling=sample)


    if input_string:
        break
pass
