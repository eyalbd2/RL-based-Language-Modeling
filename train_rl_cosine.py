

import os
import random
import logging
import numpy as np
from tensorboardX import SummaryWriter
from libbots import data, model, utils
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse

from model_test import run_test_cosine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-SAVES_DIR', type=str, default='saves', help='Save directory')
    parser.add_argument('-name', type=str, default='RL_COSINE', help='Specific model saves directory')
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

    # Load pre-trained seq2seq net
    rev_emb_dict = {idx: word for word, idx in emb_dict.items()}
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    cos_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                                hid_size=model.HIDDEN_STATE_SIZE).to(device)
    net.load_state_dict(torch.load(args.load_seq2seq_path))
    cos_net.load_state_dict(torch.load(args.load_seq2seq_path))

    writer = SummaryWriter(comment="-" + args.name)

    # BEGIN & END tokens
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)
    end_token = emb_dict[data.END_TOKEN]


    optimiser = optim.Adam(cos_net.parameters(), lr=args.LEARNING_RATE, eps=1e-3)
    batch_idx = 0
    best_cosine = None
    for epoch in range(args.MAX_EPOCHES):
        random.shuffle(train_data)
        dial_shown = False

        total_samples = 0
        skipped_samples = 0
        cosine_argmax = []
        cosine_sample = []

        for batch in data.iterate_batches(train_data, args.BATCH_SIZE):
            batch_idx += 1
            optimiser.zero_grad()
            input_seq, input_batch, output_batch = model.pack_batch_no_out(batch, cos_net.emb, device)
            enc = cos_net.encode(input_seq)

            net_policies = []
            net_actions = []
            net_advantages = []
            beg_embedding = cos_net.emb(beg_token)

            for idx, inp_idx in enumerate(input_batch):
                total_samples += 1
                ref_indices = [indices[1:] for indices in output_batch[idx]]
                item_enc = cos_net.get_encoded_item(enc, idx)
                r_argmax, actions = cos_net.decode_chain_argmax(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                stop_at_token=end_token)
                mean_emb_max = net.get_mean_emb(beg_embedding, actions)
                mean_emb_ref_list = []
                for iRef in ref_indices:
                    mean_emb_ref_list.append(net.get_mean_emb(beg_embedding, iRef))
                mean_emb_ref = sum(mean_emb_ref_list)/len(mean_emb_ref_list)
                argmax_cosine = utils.calc_cosine_many(mean_emb_max, mean_emb_ref)
                cosine_argmax.append(float(argmax_cosine))


                if not dial_shown:
                    log.info("Input: %s", utils.untokenize(data.decode_words(inp_idx, rev_emb_dict)))
                    ref_words = [utils.untokenize(data.decode_words(ref, rev_emb_dict)) for ref in ref_indices]
                    log.info("Refer: %s", " ~~|~~ ".join(ref_words))
                    log.info("Argmax: %s, cosine=%.4f",
                             utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                             argmax_cosine)

                for _ in range(args.num_of_samples):
                    r_sample, actions = cos_net.decode_chain_sampling(item_enc, beg_embedding, data.MAX_TOKENS,
                                                                                    stop_at_token=end_token)
                    mean_emb_samp = net.get_mean_emb(beg_embedding, actions)
                    sample_cosine = utils.calc_cosine_many(mean_emb_samp, mean_emb_ref)

                    if not dial_shown:
                        log.info("Sample: %s, cosine=%.4f",
                                 utils.untokenize(data.decode_words(actions, rev_emb_dict)),
                                 sample_cosine)

                    net_policies.append(r_sample)
                    net_actions.extend(actions)
                    net_advantages.extend([float(sample_cosine) - argmax_cosine] * len(actions))
                    cosine_sample.append(float(sample_cosine))
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

        cosine_test = run_test_cosine(test_data, cos_net, net, beg_token, end_token, device="cuda")
        cosine = np.mean(cosine_argmax)
        writer.add_scalar("cosine_test", cosine_test, batch_idx)
        writer.add_scalar("cosine_argmax", cosine, batch_idx)
        writer.add_scalar("cosine_sample", np.mean(cosine_sample), batch_idx)
        writer.add_scalar("epoch", batch_idx, epoch)
        log.info("Epoch %d, test COSINE: %.3f", epoch, cosine_test)
        if best_cosine is None or best_cosine < cosine_test:
            best_cosine = cosine_test
            log.info("Best cosine updated: %.4f", cosine_test)
            torch.save(cos_net.state_dict(), os.path.join(saves_path, "cosine_%.3f_%02d.dat" % (cosine_test, epoch)))
        if epoch % 10 == 0:
            torch.save(cos_net.state_dict(),
                       os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" % (epoch, cosine, cosine_test)))

    writer.close()