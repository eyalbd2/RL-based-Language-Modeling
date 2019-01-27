
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-SAVES_DIR', type=str, default='saves', help='Save directory')
    parser.add_argument('-name', type=str, default='seq2seq', help='Specific model saves directory')
    parser.add_argument('-BATCH_SIZE', type=int, default=32, help='Batch Size for training')
    parser.add_argument('-LEARNING_RATE', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('-MAX_EPOCHES', type=int, default=100, help='Number of training iterations')
    parser.add_argument('-TEACHER_PROB', type=float, default=0.5, help='Probability to force reference inputs')
    parser.add_argument('-data', type=str, default='comedy', help='Genre to use - for data')
    parser.add_argument('-train_backward', type=bool, default=False, help='Choose - train backward/forward model')
    args = parser.parse_args()

    saves_path = os.path.join(args.SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)

    log = logging.getLogger("train")
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)

    phrase_pairs, emb_dict = data.load_data(genre_filter=args.data)
    data.save_emb_dict(saves_path, emb_dict)
    end_token = emb_dict[data.END_TOKEN]
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data)

    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    log.info("Training data converted, got %d samples", len(train_data))
    log.info("Train set has %d phrases, test %d", len(train_data), len(test_data))

    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)

    writer = SummaryWriter(comment="-" + args.name)

    optimiser = optim.Adam(net.parameters(), lr=args.LEARNING_RATE)
    best_bleu = None
    for epoch in range(args.MAX_EPOCHES):
        losses = []
        bleu_sum = 0.0
        bleu_count = 0
        for batch in data.iterate_batches(train_data, args.BATCH_SIZE):
            optimiser.zero_grad()
            if args.train_backward:
                input_seq, out_seq_list, _, out_idx = model.pack_backward_batch(batch, net.emb, device)
            else:
                input_seq, out_seq_list, _, out_idx = model.pack_batch(batch, net.emb, device)
            enc = net.encode(input_seq)

            net_results = []
            net_targets = []
            for idx, out_seq in enumerate(out_seq_list):
                ref_indices = out_idx[idx][1:]
                enc_item = net.get_encoded_item(enc, idx)
                if random.random() < args.TEACHER_PROB:
                    r = net.decode_teacher(enc_item, out_seq)
                    bleu_sum += model.seq_bleu(r, ref_indices)
                else:
                    r, seq = net.decode_chain_argmax(enc_item, out_seq.data[0:1],
                                                     len(ref_indices))
                    bleu_sum += utils.calc_bleu(seq, ref_indices)
                net_results.append(r)
                net_targets.extend(ref_indices)
                bleu_count += 1
            results_v = torch.cat(net_results)
            targets_v = torch.LongTensor(net_targets).to(device)
            loss_v = F.cross_entropy(results_v, targets_v)
            loss_v.backward()
            optimiser.step()

            losses.append(loss_v.item())
        bleu = bleu_sum / bleu_count
        bleu_test = run_test(test_data, net, end_token, device)
        log.info("Epoch %d: mean loss %.3f, mean BLEU %.3f, test BLEU %.3f",
                 epoch, np.mean(losses), bleu, bleu_test)
        writer.add_scalar("loss", np.mean(losses), epoch)
        writer.add_scalar("bleu", bleu, epoch)
        writer.add_scalar("bleu_test", bleu_test, epoch)
        if best_bleu is None or best_bleu < bleu_test:
            if best_bleu is not None:
                out_name = os.path.join(saves_path, "pre_bleu_%.3f_%02d.dat" %
                                        (bleu_test, epoch))
                torch.save(net.state_dict(), out_name)
                log.info("Best BLEU updated %.3f", bleu_test)
            best_bleu = bleu_test

        if epoch % 10 == 0:
            out_name = os.path.join(saves_path, "epoch_%03d_%.3f_%.3f.dat" %
                                    (epoch, bleu, bleu_test))
            torch.save(net.state_dict(), out_name)

    writer.close()
