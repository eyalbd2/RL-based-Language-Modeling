

import os
import random
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse

from libbots import data, model, utils
from model_test import run_test_mutual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-SAVES_DIR', type=str, default='saves', help='Save directory')
    parser.add_argument('-name', type=str, default='RL_Mutual', help='Specific model saves directory')
    parser.add_argument('-BATCH_SIZE', type=int, default=32, help='Batch Size for training')
    parser.add_argument('-LEARNING_RATE', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('-MAX_EPOCHES', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('-CROSS_ENT_PROB', type=float, default=0.3, help='Probability to run a CE batch')
    parser.add_argument('-TEACHER_PROB', type=float, default=0.8, help='Probability to run an imitation batch in case '
                                                                       'of using CE')
    parser.add_argument('-data', type=str, default='comedy', help='Genre to use - for data')
    parser.add_argument('-num_of_samples', type=int, default=4, help='Number of samples per per each example')
    parser.add_argument('-load_seq2seq_path', type=str, default='Final_Saves/seq2seq/epoch_090_0.800_0.107.dat',
                        help='Pre-trained seq2seq model location')
    parser.add_argument('-laod_b_seq2seq_path', type=str, default='Final_Saves/backward_seq2seq/epoch_080_0.780_0.104.dat',
                        help='Pre-trained backward seq2seq model location')
    args = parser.parse_args()

    saves_path = os.path.join(args.SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    log = logging.getLogger("train")
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    phrase_pairs, emb_dict = data.load_data(genre_filter=args.data)
    data.save_emb_dict(saves_path, emb_dict)
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data)

    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    log.info("Training data converted, got %d samples", len(train_data))


    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}

    # Load pre-trained nets
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    net.load_state_dict(torch.load(args.load_seq2seq_path))
    back_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    back_net.load_state_dict(torch.load(args.laod_b_seq2seq_path))

    rl_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    rl_net.load_state_dict(torch.load(args.load_seq2seq_path))

    writer = SummaryWriter(comment="-" + args.name)

    # BEGIN & END tokens
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)
    end_token = emb_dict[data.END_TOKEN]



    optimiser = optim.Adam(rl_net.parameters(), lr=args.LEARNING_RATE, eps=1e-3)
    batch_idx = 0
    best_mutual = None
    for epoch in range(args.MAX_EPOCHES):
        dial_shown = False
        random.shuffle(train_data)

        total_samples = 0
        mutuals_argmax = []
        mutuals_sample = []

        for batch in data.iterate_batches(train_data, args.BATCH_SIZE):
            batch_idx += 1
            optimiser.zero_grad()
            input_seq, out_seq_list, input_batch, output_batch = model.pack_batch(batch, rl_net.emb, device)
            enc = rl_net.encode(input_seq)

            net_policies = []
            net_actions = []
            net_advantages = []
            beg_embedding = rl_net.emb(beg_token)

            if random.random() < args.CROSS_ENT_PROB:
                net_results = []
                net_targets = []
                for idx, out_seq in enumerate(out_seq_list):
                    ref_indices = output_batch[idx][1:]
                    enc_item = rl_net.get_encoded_item(enc, idx)
                    if random.random() < args.TEACHER_PROB:
                        r = rl_net.decode_teacher(enc_item, out_seq)
                    else:
                        r, seq = rl_net.decode_chain_argmax(enc_item, out_seq.data[0:1],
                                                         len(ref_indices))
                    net_results.append(r)
                    net_targets.extend(ref_indices)
                results_v = torch.cat(net_results)
                targets_v = torch.LongTensor(net_targets).to(device)
                loss_v = F.cross_entropy(results_v, targets_v)
                loss_v.backward()
                for param in rl_net.parameters():
                    param.grad.data.clamp_(-0.2, 0.2)
                optimiser.step()
            else:
                for idx, inp_idx in enumerate(input_batch):
                    total_samples += 1
                    ref_indices = output_batch[idx][1:]
                    item_enc = rl_net.get_encoded_item(enc, idx)
                    r_argmax, actions = rl_net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                stop_at_token=end_token)
                    argmax_mutual = utils.calc_mutual(net, back_net, inp_idx, actions)
                    mutuals_argmax.append(argmax_mutual)

                    if not dial_shown:
                        log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, rev_emb_dict)))
                        ref_words = [utils.untokenize(data.decode_words([ref], rev_emb_dict)) for ref in ref_indices]
                        log.info("Refer: %s", " ".join(ref_words))
                        log.info("Argmax: %s, mutual=%.4f",
                                 utils.untokenize(data.decode_words(actions, rev_emb_dict)), argmax_mutual)

                    for _ in range(args.num_of_samples):
                        r_sample, actions = rl_net.decode_chain_sampling(item_enc, beg_embedding,
                                                                      data.MAX_TOKENS, stop_at_token=end_token)
                        sample_mutual = utils.calc_mutual(net, back_net, inp_idx, actions)
                        if not dial_shown:
                            log.info("Sample: %s, mutual=%.4f",
                                     utils.untokenize(data.decode_words(actions, rev_emb_dict)), sample_mutual)

                        net_policies.append(r_sample)
                        net_actions.extend(actions)
                        net_advantages.extend([sample_mutual - argmax_mutual] * len(actions))
                        mutuals_sample.append(sample_mutual)
                    dial_shown = True

                if not net_policies:
                    continue

                policies_v = torch.cat(net_policies)
                actions_t = torch.LongTensor(net_actions).to(device)
                adv_v = torch.FloatTensor(net_advantages).to(device)
                log_prob_v = F.log_softmax(policies_v, dim=1)
                log_prob_actions_v = adv_v * log_prob_v[range(len(net_actions)), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                loss_v = loss_policy_v
                loss_v.backward()
                for param in rl_net.parameters():
                    param.grad.data.clamp_(-0.2, 0.2)
                optimiser.step()

        mutual_test = run_test_mutual(test_data, rl_net, net, back_net, beg_token, end_token, device)
        mutual = np.mean(mutuals_argmax)
        writer.add_scalar("mutual_test", mutual_test, batch_idx)
        writer.add_scalar("mutual_argmax", mutual, batch_idx)
        writer.add_scalar("mutual_sample", np.mean(mutuals_sample), batch_idx)
        writer.add_scalar("epoch", batch_idx, epoch)
        log.info("Epoch %d, test mutual: %.3f", epoch, mutual_test)
        if best_mutual is None or best_mutual < mutual_test:
            best_mutual = mutual_test
            log.info("Best mutual updated: %.4f", best_mutual)
            torch.save(rl_net.state_dict(), os.path.join(saves_path, "mutual_%.3f_%02d.dat" % (mutual_test, epoch)))
        if epoch % 10 == 0:
            torch.save(rl_net.state_dict(),
                       os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" % (epoch, mutual, mutual_test)))

    writer.close()