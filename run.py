from model import *
import utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy


def train(data, params):
    # load word2vec
    print("loading word2vec...")
    word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    wv_matrix = []
    for i in range(len(data["vocab"])):
        word = data["idx_to_word"][i]

        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

    # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    wv_matrix.append(np.zeros(300).astype("float32"))
    wv_matrix = np.array(wv_matrix*3)
    params["WV_MATRIX"] = wv_matrix
    params["IN_CHANNEL"] = len(wv_matrix)

    model1 = AttentionCNN(**params).cuda(params["GPU"])
    model2 = AttentionRNN(**params).cuda(params['GPU'])
    model3 = FullConnect(**params).cuda(params['GPU'])

    parameters1 = filter(lambda p: p.requires_grad, model1.parameters())
    parameters2 = filter(lambda p: p.requires_grad, model2.parameters())
    parameters3 = filter(lambda p: p.requires_grad, model3.parameters())
    optimizer1 = optim.Adadelta(parameters1, params["LEARNING_RATE"])
    optimizer2 = optim.Adadelta(parameters2, params["LEARNING_RATE"])
    optimizer3 = optim.Adadelta(parameters3, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

            state_gram = model2.init_hidden().cuda(params["GPU"])

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            model1.train()
            model2.train()
            model3.train()
            _conv, att_local = model1(batch_x)
            att_global = model2(_conv, state_gram)
            pred = model3(att_local, att_global)
            loss = criterion(pred.cuda(params["GPU"]), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters1, max_norm=params["NORM_LIMIT"])
            nn.utils.clip_grad_norm(parameters2, max_norm=params["NORM_LIMIT"])
            nn.utils.clip_grad_norm(parameters3, max_norm=params["NORM_LIMIT"])
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

        dev_acc = test(data, model1, model2, model3, params, mode="dev")
        test_acc = test(data, model1, model2, model3, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model1 = copy.deepcopy(model1)
            best_model2 = copy.deepcopy(model2)
            best_model3 = copy.deepcopy(model3)

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model1, best_model2, best_model3


def test(data, model1, model2, model3, params, mode="test"):
    model1.eval()
    model2.eval()
    model3.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    state_gram = model2.init_hidden().cuda(params["GPU"])

    _conv, att_local = model1(x)
    att_global = model2(_conv, state_gram)
    pred = np.argmax(model3(att_local, att_global).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-RNN-Attention-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used")
    parser.add_argument("--filter_num", default=100, type=int, help="number of CNN filters to be used")
    parser.add_argument("--filter_size", default=5, type=int, help="the size of the filter")
    parser.add_argument("--final_dim", default=150, type=int, help="the output dim of CNN and RNN")
    parser.add_argument("--hidden_size", default=150, type=int, help="the size of hidden layer")
    parser.add_argument("--blance_val", default=0.5, type=float, help="the attention value of CNN output value")

    options = parser.parse_args()
    data = getattr(utils, f"read_{options.dataset}")()

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "NUM_CLASS": len(data["classes"]),
        "FILTER_SIZE": options.filter_size,
        "FILTER_NUM": options.filter_num,
        "DROPOUT_RATE": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu,
        "HIDDEN_SIZE": options.hidden_size,
        "BLANCE_VAL": options.blance_val,
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model1, model2, model3 = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model1, params, name="model1")
            utils.save_model(model2, params, name="model2")
            utils.save_model(model3, params, name="model3")
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model1 = utils.load_model(params, name="model1").cuda(params["GPU"])
        model2 = utils.load_model(params, name="model2").cuda(params["GPU"])
        model3 = utils.load_model(params, name="model3").cuda(params["GPU"])
        test_acc = test(data, model1, model2, model3, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()
