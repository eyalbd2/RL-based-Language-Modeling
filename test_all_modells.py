
import os
import logging
import numpy as np
import torch
import argparse

from libbots import data, model
from model_test import run_test, run_test_mutual, run_test_preplexity, run_test_cosine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-SAVES_DIR', type=str, default='saves', help='Save directory')
    parser.add_argument('-name', type=str, default='RL_BLUE', help='Specific model saves directory')
    parser.add_argument('-BATCH_SIZE', type=int, default=32, help='Batch Size for training')
    parser.add_argument('-data', type=str, default='comedy', help='Genre to use - for data')
    parser.add_argument('-num_of_samples', type=int, default=4, help='Number of samples per per each example')
    parser.add_argument('-load_seq2seq_path', type=str, default='Final_Saves/seq2seq/epoch_090_0.800_0.107.dat',
                        help='Pre-trained seq2seq model location')
    parser.add_argument('-laod_b_seq2seq_path', type=str,
                        default='Final_Saves/backward_seq2seq/epoch_080_0.780_0.104.dat',
                        help='Pre-trained backward seq2seq model location')
    parser.add_argument('-bleu_model_path', type=str, default='Final_Saves/RL_BLUE/bleu_0.135_177.dat',
                        help='Pre-trained BLEU model location')
    parser.add_argument('-mutual_model_path', type=str, default='Final_Saves/RL_Mutual/epoch_180_-4.325_-7.192.dat',
                        help='Pre-trained MMI model location')
    parser.add_argument('-prep_model_path', type=str, default='Final_Saves/RL_Perplexity/epoch_050_1.463_3.701.dat',
                        help='Pre-trained Perplexity model location')
    parser.add_argument('-cos_model_path', type=str, default='Final_Saves/RL_COSINE/cosine_0.621_03.dat',
                        help='Pre-trained Cosine Similarity model location')
    args = parser.parse_args()


    log = logging.getLogger("test")

    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    arg_data = 'comedy'

    # Load Seq2Seq
    seq2seq_emb_dict = data.load_emb_dict(os.path.dirname(args.load_seq2seq_path))
    seq2seq_rev_emb_dict = {idx: word for word, idx in seq2seq_emb_dict.items()}
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(seq2seq_emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    net.load_state_dict(torch.load(args.load_seq2seq_path))

    # Load Back Seq2Seq
    b_seq2seq_emb_dict = data.load_emb_dict(os.path.dirname(args.laod_b_seq2seq_path))
    b_seq2seq_rev_emb_dict = {idx: word for word, idx in b_seq2seq_emb_dict.items()}
    b_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(b_seq2seq_emb_dict),
                              hid_size=model.HIDDEN_STATE_SIZE).to(device)
    b_net.load_state_dict(torch.load(args.laod_b_seq2seq_path))

    # Load BLEU
    bleu_emb_dict = data.load_emb_dict(os.path.dirname(args.bleu_model_path))
    bleu_rev_emb_dict = {idx: word for word, idx in bleu_emb_dict.items()}
    bleu_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(bleu_emb_dict),
                                 hid_size=model.HIDDEN_STATE_SIZE).to(device)
    bleu_net.load_state_dict(torch.load(args.bleu_model_path))

    # Load Mutual
    mutual_emb_dict = data.load_emb_dict(os.path.dirname(args.mutual_model_path))
    mutual_rev_emb_dict = {idx: word for word, idx in mutual_emb_dict.items()}
    mutual_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(mutual_emb_dict),
                                   hid_size=model.HIDDEN_STATE_SIZE).to(device)
    mutual_net.load_state_dict(torch.load(args.mutual_model_path))


    # Load Preplexity
    prep_emb_dict = data.load_emb_dict(os.path.dirname(args.prep_model_path))
    prep_rev_emb_dict = {idx: word for word, idx in prep_emb_dict.items()}
    prep_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(prep_emb_dict),
                                 hid_size=model.HIDDEN_STATE_SIZE).to(device)
    prep_net.load_state_dict(torch.load(args.prep_model_path))


    # Load Cosine Similarity
    cos_emb_dict = data.load_emb_dict(os.path.dirname(args.cos_model_path))
    cos_rev_emb_dict = {idx: word for word, idx in cos_emb_dict.items()}
    cos_net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(cos_emb_dict),
                                hid_size=model.HIDDEN_STATE_SIZE).to(device)
    cos_net.load_state_dict(torch.load(args.cos_model_path))


    phrase_pairs, emb_dict = data.load_data(genre_filter=arg_data)
    end_token = emb_dict[data.END_TOKEN]
    train_data = data.encode_phrase_pairs(phrase_pairs, emb_dict)
    rand = np.random.RandomState(data.SHUFFLE_SEED)
    rand.shuffle(train_data)
    train_data, test_data = data.split_train_test(train_data)

    # BEGIN token
    beg_token = torch.LongTensor([emb_dict[data.BEGIN_TOKEN]]).to(device)

    # # Test Seq2Seq model
    bleu_test_seq2seq = run_test(test_data, net, end_token, device)
    mutual_test_seq2seq = run_test_mutual(test_data, net, net, b_net, beg_token, end_token, device)
    preplexity_test_seq2seq = run_test_preplexity(test_data, net, net, end_token, device)
    cosine_test_seq2seq = run_test_cosine(test_data, net, beg_token, end_token, device)
    #
    # # Test BLEU model
    bleu_test_bleu = run_test(test_data, bleu_net, end_token, device)
    mutual_test_bleu = run_test_mutual(test_data, bleu_net, net, b_net, beg_token, end_token, device)
    preplexity_test_bleu = run_test_preplexity(test_data, bleu_net, net, end_token, device)
    cosine_test_bleu = run_test_cosine(test_data, bleu_net, beg_token, end_token, device)

    # # Test Mutual Information model
    bleu_test_mutual = run_test(test_data, mutual_net, end_token, device)
    mutual_test_mutual = run_test_mutual(test_data, mutual_net, net, b_net, beg_token, end_token, device)
    preplexity_test_mutual = run_test_preplexity(test_data, mutual_net, net, end_token, device)
    cosine_test_mutual = run_test_cosine(test_data, mutual_net, beg_token, end_token, device)

    # Test Perplexity model
    bleu_test_per = run_test(test_data, prep_net, end_token, device)
    mutual_test_per = run_test_mutual(test_data, prep_net, net, b_net, beg_token, end_token, device)
    preplexity_test_per = run_test_preplexity(test_data, prep_net, net, end_token, device)
    cosine_test_per = run_test_cosine(test_data, prep_net, beg_token, end_token, device)

    # Test Cosine Similarity model
    bleu_test_cos = run_test(test_data, cos_net, end_token, device)
    mutual_test_cos = run_test_mutual(test_data, cos_net, net, b_net, beg_token, end_token, device)
    preplexity_test_cos = run_test_preplexity(test_data, cos_net, net, end_token, device)
    cosine_test_cos = run_test_cosine(test_data, cos_net, beg_token, end_token, device)

    log.info("Obtained %d phrase pairs with %d uniq words", len(phrase_pairs), len(emb_dict))
    log.info("-----------------------------------------------")
    log.info("BLEU scores:")
    log.info("    Seq2Seq -             %.3f", bleu_test_seq2seq)
    log.info("    BLEU -                %.3f", bleu_test_bleu)
    log.info("    MMI -                 %.3f", bleu_test_mutual)
    log.info("    Perplexity -          %.3f", bleu_test_per)
    log.info("    Cosine Similatirity - %.3f", bleu_test_cos)
    log.info("-----------------------------------------------")
    log.info("Max Mutual Information scores:")
    log.info("    Seq2Seq -             %.3f", mutual_test_seq2seq)
    log.info("    BLEU -                %.3f", mutual_test_bleu)
    log.info("    MMI -                 %.3f", mutual_test_mutual)
    log.info("    Perplexity -          %.3f", preplexity_test_mutual)
    log.info("    Cosine Similatirity - %.3f", mutual_test_cos)
    log.info("-----------------------------------------------")
    log.info("Perplexity scores:")
    log.info("    Seq2Seq -             %.3f", preplexity_test_seq2seq)
    log.info("    BLEU -                %.3f", preplexity_test_bleu)
    log.info("    MMI -                 %.3f", preplexity_test_mutual)
    log.info("    Perplexity -          %.3f", preplexity_test_per)
    log.info("    Cosine Similatirity - %.3f", preplexity_test_cos)
    log.info("-----------------------------------------------")
    log.info("Cosine Similarity scores:")
    log.info("    Seq2Seq -             %.3f", cosine_test_seq2seq)
    log.info("    BLEU -                %.3f", cosine_test_bleu)
    log.info("    MMI -                 %.3f", mutual_test_cos)
    log.info("    Perplexity -          %.3f", cosine_test_per)
    log.info("    Cosine Similatirity - %.3f", cosine_test_cos)
    log.info("-----------------------------------------------")

