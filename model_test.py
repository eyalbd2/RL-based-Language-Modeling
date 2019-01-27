
from libbots import model, utils, data

def run_test(test_data, per_net, end_token, device="cuda"):
    bleu_sum = 0.0
    bleu_count = 0
    for p1, p2 in test_data:
        input_seq = model.pack_input(p1, per_net.emb, device)
        enc = per_net.encode(input_seq)
        _, tokens = per_net.decode_chain_argmax(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                            stop_at_token=end_token)
        ref_indices = p2[1:]
        bleu_sum += utils.calc_bleu_many(tokens, [ref_indices])
        bleu_count += 1
    return bleu_sum / bleu_count

def run_test_preplexity(test_data, per_net, net, end_token, device="cpu"):
    preplexity_sum = 0.0
    preplexity_count = 0
    for p1, p2 in test_data:
        input_seq = model.pack_input(p1, per_net.emb, device)
        enc = per_net.encode(input_seq)
        logits, tokens = per_net.decode_chain_argmax(enc, input_seq.data[0:1], seq_len=data.MAX_TOKENS,
                                            stop_at_token=end_token)
        r = net.get_logits(enc, input_seq.data[0:1], data.MAX_TOKENS, tokens,
                                          stop_at_token=end_token)
        preplexity_sum += utils.calc_preplexity_many(r, tokens)
        preplexity_count += 1
    return preplexity_sum / preplexity_count

def run_test_mutual(test_data, rl_net, net, back_net, end_token, device="cuda"):
    mutual_sum = 0.0
    mutual_count = 0
    for p1, p2 in test_data:
        input_seq_bleu = model.pack_input(p1, rl_net.emb, device)
        enc_bleu = rl_net.encode(input_seq_bleu)
        _, actions = rl_net.decode_chain_argmax(enc_bleu, input_seq_bleu.data[0:1], seq_len=data.MAX_TOKENS,
                                                stop_at_token=end_token)

        mutual_sum += utils.calc_mutual(net, back_net, p1, actions)
        mutual_count += 1
    return mutual_sum / mutual_count

def run_test_cosine(test_data, rl_net, net, beg_token, end_token, device="cuda"):
    cosine_sum = 0.0
    cosine_count = 0
    beg_embedding = rl_net.emb(beg_token)
    for p1, p2 in test_data:
        input_seq_cosine = model.pack_input(p1, rl_net.emb, device)
        enc_cosine = rl_net.encode(input_seq_cosine)
        r_argmax, actions = rl_net.decode_chain_argmax(enc_cosine, beg_embedding, data.MAX_TOKENS,
                                                        stop_at_token=end_token)
        mean_emb_max = net.get_mean_emb(beg_embedding, actions)
        mean_emb_ref_list = []
        for iRef in p2:
            mean_emb_ref_list.append(rl_net.get_mean_emb(beg_embedding, iRef))
        mean_emb_ref = sum(mean_emb_ref_list) / len(mean_emb_ref_list)
        cosine_sum += utils.calc_cosine_many(mean_emb_max, mean_emb_ref)
        cosine_count += 1
    return cosine_sum / cosine_count