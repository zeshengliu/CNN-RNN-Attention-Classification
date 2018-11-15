import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def batch_matmul_bias(seq, weight, bias, nonlinear=''):
    output = None
    bias_dim = bias.size()
    for i in range(seq.size()[0]):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
        if nonlinear == 'tanh':
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if output is None:
            output = _s_bias
        else:
            output = torch.cat((output, _s_bias), 0)
        return output.squeeze()


def batch_matmul(seq, weight, nonlinear=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinear == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
        return s.squeeze()


def attention_mul(outputs, att_weights):
    attn_vectors = None
    for i in range(outputs.size(0)):
        h_i = outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
        return torch.sum(attn_vectors, 0)


class AttentionCNN(nn.Module):
    def __init__(self, **kwargs):
        super(AttentionCNN, self).__init__()
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.FILTERS_SIZE = kwargs["FILTERS_SIZE"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.IN_CHANNEL = kwargs['IN_CHANNEL']
        self.WV_MATRIX = kwargs["WV_MATRIX"]
        self.FINAL_DIM = kwargs["FINAL_DIM"]
        # assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding

        self.embedding = [nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM,
                                       padding_idx=self.VOCAB_SIZE + 1)] * self.IN_CHANNEL

        for i in range(self.IN_CHANNEL):
            self.embedding[i].weight.data.copy_(torch.from_numpy(self.WV_MATRIX[i]))
            self.embedding[i].weight.requires_grad = False

        self.conv = nn.Conv1d(self.IN_CHANNEL, self.FILTERS_SIZE,
                              self.WORD_DIM * self.FILTERS_SIZE, stride=self.WORD_DIM)

        self.weight_W_word = nn.Parameter(torch.Tensor(self.FILTER_NUM, self.FINAL_DIM))
        self.bias_word = nn.Parameter(torch.Tensor(self.FINAL_DIM, 1))
        self.vw_local = nn.Parameter(torch.Tensor(self.FINAL_DIM, 1))

        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.vw_local.data.uniform_(-0.1, 0.1)

    def forward(self, inp):
        x = []
        for i in range(self.IN_CHANNEL):
            x.append(self.embedding[i](inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN))
        x = torch.cat(x, 1)
        conv_res = self.conv(x)
        conv_res = conv_res.transpose(1, 2)
        gram_squish = batch_matmul_bias(conv_res, self.weight_W_word, self.bias_word, nonlinear='tanh')
        gram_att = batch_matmul(gram_squish, self.vw_local)
        gram_att_norm = nn.Softmax(gram_att)
        gram_att_vec = attention_mul(gram_squish, gram_att_norm)
        return conv_res, gram_att_vec


class AttentionRNN(nn.Module):
    def __init__(self, **kwargs):
        super(AttentionRNN, self).__init__()
        self.batch_size = kwargs["BATCH_SIZE"]
        self.final_dim = kwargs["FINAL_DIM"]
        self.input_dim = kwargs["FINAL_DIM"]

        self.lstm = nn.LSTM(2*self.input_dim, self.final_dim, bidirectional=True)
        self.gram_se_mx = nn.Parameter(torch.Tensor(2*self.final_dim, self.final_dim))
        self.gram_bias = nn.Parameter(torch.Tensor(self.final_dim, 1))
        self.vw_global = nn.Parameter(torch.Tensor(self.final_dim, 1))

        self.gram_se_mx.data.uniform_(-0.1, 0.1)
        self.vw_global.data.uniform_(-0.1, 0.1)

    def forward(self, conv_res, state_gram):
        output_vec, state_gram = self.lstm(conv_res, state_gram)
        output_squish = batch_matmul_bias(output_vec, self.gram_se_mx, self.gram_bias, nonlinear='tanh')
        output_att = batch_matmul(output_squish, self.vw_global)
        output_att_norm = nn.Softmax(output_att)
        output_att_vec = attention_mul(output_squish, output_att_norm)
        return output_att_vec

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.final_dim))


class FullConnect(nn.Module):
    def __init__(self, **kwargs):
        super(FullConnect, self).__init__()
        self.input_size = kwargs["FINAL_DIM"]*2
        self.hidden_size = kwargs["HIDDEN_SIZE"]
        self.num_class = kwargs["NUM_CLASS"]
        self.dropout_rate = kwargs["DROPOUT_RATE"]
        self.blance_val = kwargs["BLANCE_VAL"]
        self.linear1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size), self.nn.ReLU(True))
        self.linear2 = nn.Linear(self.hidden_size, self.num_class)

    def forward(self, cnn_att_vec, rnn_att_vec):
        lamda_val = Variable(torch.Tensor([self.lamda_val]), requires_grad=True)
        x1 = lamda_val*cnn_att_vec
        x2 = (1-lamda_val)*rnn_att_vec
        x = torch.cat(x1, x2, 1)
        x = self.linear1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear2(x)
        x = F.softmax(x, dim=1)
        return x
