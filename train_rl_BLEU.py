

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
from model_test import run_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-SAVES_DIR', type=str, default='saves', help='Save directory')
    parser.add_argument('-name', type=str, default='RL_BLUE', help='Specific model saves directory')
    parser.add_argument('-BATCH_SIZE', type=int, default=16, help='Batch Size for training')
    parser.add_argument('-LEARNING_RATE', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('-MAX_EPOCHES', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('-data', type=str, default='comedy', help='Genre to use - for data')
    parser.add_argument('-num_of_samples', type=int, default=4, help='Number of samples per per each example')
    parser.add_argument('-load_seq2seq_path', type=str, default='Final_Saves/seq2seq/epoch_090_0.800_0.107.dat',
                        help='Pre-trained seq2seq model location')
    args = parser.parse_args()

    saves_path = os.path.join(args.SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)


    log = logging.getLogger("train")
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phrase_pairs, emb_dict = data.load_data(genre_filter=args.data)
    data.save_emb_dict(saves_path, emb_dict)
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data)
    train_data = data.group_train_data(train_data)
    test_data = data.group_train_data(test_data)

    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    log.info("Training data converted, got %d samples", len(train_data))
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))

    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    loaded_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)

    writer = SummaryWriter(comment="-" + args.name)
    net.load_state_dict(torch.load(args.load_seq2seq_path))

    # BEGIN & END tokens
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)
    end_token = emb_dict[data.END_TOKEN]


    optimiser = optim.Adam(net.parameters(), lr=args.LEARNING_RATE, eps=1e-3)
    batch_idx = 0
    best_bleu = None
    for epoch in range(args.MAX_EPOCHES):
        random.shuffle(train_data)
        dial_shown = False

        total_samples = 0
        bleus_argmax = []
        bleus_sample = []

        for batch in data.iterate_batches(train_data, args.BATCH_SIZE):
            batch_idx += 1
            optimiser.zero_grad()
            input_seq, input_batch, output_batch = model.pack_batch_no_out(batch, net.emb, device)
            enc = net.encode(input_seq)

            net_policies = []
            net_actions = []
            net_advantages = []
            beg_embedding = net.emb(beg_token)

            for idx, inp_idx in enumerate(input_batch):
                total_samples += 1
                ref_indices = [indices[1:] for indices in output_batch[idx]]
                item_enc = net.get_encoded_item(enc, idx)
                r_argmax, actions = net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS,
                                                            stop_at_token=end_token)
                argmax_bleu = utils.calc_bleu_many(actions, ref_indices)
                bleus_argmax.append(argmax_bleu)


                if not dial_shown:
                    log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, rev_emb_dict)))
                    ref_words = [utils.untokenize(data.decode_words(ref, rev_emb_dict)) for ref in ref_indices]
                    log.info("Refer: %s", " ~~|~~ ".join(ref_words))
                    log.info("Argmax: %s, bleu=%.4f",
                             utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                             argmax_bleu)

                for _ in range(args.num_of_samples):
                    r_sample, actions = net.decode_chain_sampling(item_enc, beg_embedding,
                                                                  data.MAX_TOKENS, stop_at_token=end_token)
                    sample_bleu = utils.calc_bleu_many(actions, ref_indices)

                    if not dial_shown:
                        log.info("Sample: %s, bleu=%.4f",
                                 utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                                 sample_bleu)

                    net_policies.append(r_sample)
                    net_actions.extend(actions)
                    net_advantages.extend([sample_bleu - argmax_bleu] * len(actions))
                    bleus_sample.append(sample_bleu)
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
            optimiser.step()

        bleu_test = run_test(test_data, net, end_token, device)
        bleu = np.mean(bleus_argmax)
        writer.add_scalar("bleu_test", bleu_test, batch_idx)
        writer.add_scalar("bleu_argmax", bleu, batch_idx)
        writer.add_scalar("bleu_sample", np.mean(bleus_sample), batch_idx)
        writer.add_scalar("epoch", batch_idx, epoch)

        log.info("Epoch %d, test BLEU: %.3f", epoch, bleu_test)
        if best_bleu is None or best_bleu < bleu_test:
            best_bleu = bleu_test
            log.info("Best bleu updated: %.4f", bleu_test)
            torch.save(net.state_dict(), os.path.join(saves_path, "bleu_%.3f_%02d.dat" % (bleu_test, epoch)))
        if epoch % 10 == 0:
            torch.save(net.state_dict(),
                       os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" % (epoch, bleu, bleu_test)))

    writer.close()