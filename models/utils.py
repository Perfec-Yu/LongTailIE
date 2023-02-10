import torch
import torch.nn as nn
from typing import Callable, Dict, Optional, Union, List
import math
from functools import reduce
import torch.nn.functional as F
from .bert_layer_utils import surrogate_pre_forward, surrogate_post_forward


class SimpleCRFHead(nn.Module):
    def __init__(self, nstate:int=3, trainable:bool=True, bio=False):
        super().__init__()
        self.tran = nn.Parameter(torch.zeros(nstate, nstate), requires_grad=trainable)
        self.init = nn.Parameter(torch.zeros(nstate), requires_grad=trainable)
        self.mask = nn.Parameter(torch.zeros(nstate, nstate), requires_grad=False)
        self.imask = nn.Parameter(torch.zeros(nstate), requires_grad=False)
        if bio:
            for i in range(nstate):
                if i > 0 and i % 2 == 0:
                    self.imask[i] = float("-inf")
                for j in range(nstate):
                    if i == 0 :
                        if j > 0 and j % 2 == 0: # O I
                            self.mask[i, j] = float("-inf")
                    elif i % 2 == 1: # B-a I-b
                        if j > 0 and j % 2 == 0 and j != i + 1:
                            self.mask[i, j] = float("-inf")
                    elif i % 2 == 0: # I-a I-b
                        if j > 0 and j % 2 == 0 and j != i:
                            self.mask[i, j] = float("-inf")
        self.nstate = nstate

    def forward(self, inputs:torch.FloatTensor, path:torch.LongTensor, seq_mask:Optional[torch.BoolTensor]=None):
        '''
        inputs: * x L x nstate
        path: * x L
        seq_mask: * x L
        '''
        sizes = path.size()
        path_scores = torch.gather(inputs, -1, path.unsqueeze(-1)).view(-1, sizes[-1])
        tran_scores = self.tran[path[..., :-1], path[..., 1:]].view(-1, sizes[-1]-1)
        init_scores = self.init[path[..., 0]].view(-1)
        if seq_mask is None:
            path_scores = torch.sum(path_scores, dim=-1) + init_scores + torch.sum(tran_scores, dim=-1)
        else:
            path_masks = seq_mask.view(-1, sizes[-1]).float()
            tran_masks = torch.logical_and(seq_mask[..., :-1], seq_mask[..., 1:]).view(-1, sizes[-1]-1).float()
            path_scores = torch.sum(path_scores * path_masks, dim=-1) + init_scores + torch.sum(tran_scores*  tran_masks, dim=-1)
            seq_mask[..., :-1] = torch.logical_and(seq_mask[..., :-1], ~seq_mask[..., 1:])

        path_scores = path_scores.view(sizes[:-1])

        previous = torch.zeros_like(inputs[..., 0, :]).unsqueeze(-1) + (self.init + self.imask).view(*([1]*(len(sizes)-1)+[self.nstate, 1]))
        previous = previous + inputs[..., 0, :].unsqueeze(-1)
        tran = (self.tran + self.mask).view(*([1]*(len(sizes)-1)+[self.nstate, self.nstate]))
        scores = torch.zeros_like(inputs[..., 0, 0]).detach()
        for step in range(1, sizes[-1]):
            previous = previous + tran + inputs[..., step, :].unsqueeze(-2)
            previous = torch.logsumexp(previous, dim=-2)
            if seq_mask is not None and torch.any(seq_mask[..., step]):
                scores[seq_mask[..., step]] = torch.logsumexp(previous[seq_mask[..., step]], dim=-1)
            previous = previous.unsqueeze(-1)
        if seq_mask is None:
            scores = torch.logsumexp(previous, dim=[-2,-1])
        # if torch.any(scores < path_scores):
        #     print(scores, path_scores, scores-path_scores, path, seq_mask)
        #     input()
        return scores - path_scores

    def prediction(self, inputs:torch.FloatTensor):
        '''
        inputs: * x L x nstate
        '''
        states = inputs[..., 0, :]
        tran = (self.tran+self.mask).view(*([1]*(len(inputs.size())-2)+[self.nstate, self.nstate]))
        init = (self.init+self.imask).view(*([1]*(len(inputs.size())-2)+[self.nstate]))
        states = states + init
        path = torch.zeros_like(inputs).long()

        for step in range(1, inputs.size(-2)):
            next_states = inputs[..., step, :].unsqueeze(-2) + tran + states.unsqueeze(-1)
            states, index = torch.max(next_states, dim=-2)
            if step > 1:
                path = torch.gather(path, -1, index.repeat_interleave(inputs.size(-2), -1).view(*inputs.size()[:-2], inputs.size(-1), inputs.size(-2)).transpose(-2, -1))
            path[..., step-1, :] = index

        score, path_index = torch.max(states, dim=-1)
        pred = torch.gather(path, -1, path_index.unsqueeze(-1).repeat_interleave(inputs.size(-2), -1).unsqueeze(-1))
        pred = pred.squeeze(-1)
        pred[..., -1] = path_index
        return pred, score


class Bilinear(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        _weight = torch.randn(out_features, in1_features, in2_features) * math.sqrt(2 / (in1_features + in2_features))
        self.weight = nn.Parameter(_weight)
        if bias:
            _bias = torch.ones(out_features) * math.sqrt(2 / (in1_features + in2_features))
            self.bias = nn.Parameter(_bias)
        else:
            self.bias = None
        self.out_features = out_features
        self.in1_features = in1_features
        self.in2_features = in2_features

    def forward(self, input1, input2):
        # B x n x d
        assert len(input1.size()) == len(input2.size())
        input_dims = len(input1.size())
        weight_size = [1] * (input_dims-2) + list(self.weight.size())
        bias_size = [1] * (input_dims-2) + [self.out_features] + [1, 1]
        weight = self.weight.view(*weight_size)
        if self.bias is not None:
            bias = self.bias.view(*bias_size)
        input1 = input1.unsqueeze(-3)
        input2 = input2.unsqueeze(-3).transpose(-2, -1)
        outputs = bias + torch.matmul(input1,
                                     torch.matmul(self.weight.unsqueeze(0),
                                                  input2))
        return outputs.permute(*list(range(0, input_dims-2)), input_dims-1, input_dims, input_dims-2)


class BiClassifier(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.in1_linear = nn.Linear(in1_features, 1024, bias=bias)
        self.in2_linear = nn.Linear(in2_features, 1024, bias=False)
        self.out_linear = nn.Linear(1024, out_features, bias=bias)
    def forward(self, input1, input2):
        in1 = self.in1_linear(input1).unsqueeze(-2)
        in2 = self.in2_linear(input2).unsqueeze(-3)
        return self.out_linear(torch.relu(in1 + in2))


class Filters(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, kernel_size:int, dilations:Union[int, List[int]]=1, add_projection:bool=True, lam=1e-3):
        super().__init__()
        self.kernel_size = kernel_size
        if isinstance(dilations, int): dilations = [dilations]
        for d in dilations: assert input_dim % (d * kernel_size) == 0

        self.dilations = dilations

        self.indices = []
        for d in dilations:
            starting_indices = reduce(lambda x,y:x+y, [list(range(i, input_dim, kernel_size*d)) for i in range(d)])
            for start_index in starting_indices:
                self.indices.append(list(range(start_index, start_index + d * kernel_size, d)))

        n_heads = len(self.indices)
        self.classifiers = nn.ModuleList([nn.Linear(kernel_size, output_dim) for _ in range(n_heads)])
        self.n_heads = n_heads

        self.projection = nn.Linear(input_dim, input_dim, bias=False) if add_projection else None
        self.eye = nn.Parameter(torch.eye(input_dim, requires_grad=False), requires_grad=False)
        self.lam = lam


    def loss(self, xs, y):
        ps = [torch.log_softmax(x, dim=-1) for x in xs]


    def forward(self, x:torch.Tensor, y:Optional[torch.LongTensor]=None, loss_function:Optional[Callable]=None):

        if self.projection: x = self.projection(x)
        slices = [x[..., indice] for indice in self.indices]
        logits = [c(s) for c,s in zip(self.classifiers, slices)]
        if y is not None:
            if loss_function is None: loss_function = lambda logits, y: sum([F.cross_entropy(l, y) for l in logits]) / len(logits)
            loss = loss_function(logits, y)
            reg = torch.sum((torch.matmul(self.projection.weight, self.projection.weight.transpose(0, 1)) - self.eye)**2)
            return loss + self.lam * reg, logits
        else:
            return logits


class MomentumClassifier(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, nheads:int, alpha:float, gamma:float, tau:float, mu:float):
        super().__init__()
        self.classifier = nn.Parameter(torch.Tensor(output_dim, input_dim), requires_grad=True)
        stdv = 1. / math.sqrt(self.classifier.size(1))
        self.classifier.data.uniform_(-stdv, stdv)
        self.register_buffer("causal_embed", torch.FloatTensor(1, input_dim).zero_())
        self.head_dim = input_dim // nheads
        self.nheads = nheads
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.mu = mu
        self.output_dim = output_dim


    def normalize(self, x, scale:float=0.):
        return  x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-9 + scale)

    def forward(self, x, labels, training=True):
        normed_w = self.multi_head_call(self.normalize, self.classifier, nheads=self.nheads, scale=self.gamma)
        normed_x = self.multi_head_call(self.normalize, x, nheads=self.nheads)
        logits = torch.matmul(normed_x * self.tau, normed_w.transpose(0, 1))
        if training:
            self.update_embed(x, labels)
            return logits
        else:
            normed_e = self.multi_head_call(self.normalize, self.causal_embed, nheads=self.nheads)
            x_list = torch.split(normed_x, self.head_dim, dim=-1)
            e_list = torch.split(normed_e, self.head_dim, dim=-1)
            w_list = torch.split(normed_w, self.head_dim, dim=-1)
            output = [torch.matmul((self.alpha * self.cos(nx, ne.unsqueeze(0)) * ne) * self.tau, nw.transpose(0, 1)) for nx, ne, nw in zip(x_list, e_list, w_list)]
            tde_logits = sum(output)

            old_score = logits.softmax(-1)
            new_score = (logits - tde_logits).softmax(-1)

            final_score_na = old_score[..., :1].contiguous()
            final_score_pos = old_score[..., 1:].sum(-1, keepdim=True) / (new_score[..., 1:].sum(-1, keepdim=True) + 1e-9) * (new_score[..., 1:] + 1e-9 / (self.output_dim-1))
            final_score = torch.cat([final_score_na, final_score_pos], dim=-1)
            return logits, final_score

    def multi_head_call(self, func, x, nheads, **kwargs):
        x_list = torch.split(x, self.head_dim, dim=-1)
        y_list = [func(item, **kwargs) for item in x_list]
        assert len(x_list) == nheads
        assert len(y_list) == nheads
        return torch.cat(y_list, dim=-1)

    def update_embed(self, x, labels):
        with torch.no_grad():
            if torch.sum((labels>0).long()) > 0:
                pos_features = x[labels > 0].clone().detach().mean(0, keepdim=True)
                self.causal_embed = self.mu * self.causal_embed + pos_features
        if torch.any(torch.isnan(self.causal_embed)):
            raise ValueError
        return

    def cos(self, x, y):
        return (x * y).sum(-1, keepdim=True) / torch.norm(x, p=2, dim=-1, keepdim=True) / torch.norm(y, p=2, dim=-1, keepdim=True)


class SurrogateClassifier(nn.Module):
    def __init__(self, input_dim:int,  output_dim:int, mu:float, lam:float, nheads:int=2, na_att:bool=False):
        super().__init__()
        self.nheads = nheads

        self.classifier = nn.Parameter(torch.Tensor(output_dim, input_dim), requires_grad=True)
        stdv = 1. / math.sqrt(self.classifier.size(1))
        self.classifier.data.uniform_(-stdv, stdv)

        self.ctx_q = nn.Linear(input_dim, output_dim if na_att else output_dim-1)
        self.word_g = nn.Linear(input_dim, 1)

        self.output_dim = output_dim
        self.head_dim = input_dim // nheads
        self.mu = mu
        self.na_att = na_att
        self.tau = 16 / self.nheads
        self.lam = lam
        self.use_cos = False

        self.register_buffer("history_input", torch.FloatTensor(output_dim-1, input_dim).zero_())
        self.register_buffer("history_count", torch.LongTensor(output_dim-1).zero_())

    def p_loss(self, logits, plabels):
        if self.na_att:
            y_na = (torch.sum(plabels, dim=-1, keepdim=True) < 0.5).float()
            y = torch.cat((y_na, plabels), dim=-1)
            y = y / torch.sum(y, dim=-1, keepdim=True)
            if torch.any(torch.isnan(y)):
                raise ValueError
            return -torch.mean(y * torch.log_softmax(logits, dim=-1))
        else:
            return F.binary_cross_entropy_with_logits(logits, plabels)


    def forward(self, x, x_ctx, labels, training=True):
        if self.use_cos:
            normed_w = self.multi_head_call(self.normalize, self.classifier, nheads=self.nheads, scale=self.lam)
        if training:
            self.update_history(x, labels)
        r_sub, att_logits = self.generate_substitute(x_ctx)
        g_word = torch.sigmoid(self.word_g(x))
        if self.use_cos:
            normed_x = self.multi_head_call(self.normalize, (g_word * x + (1 - g_word) * r_sub), nheads=self.nheads)
            logits = torch.matmul(normed_x * self.tau, normed_w.transpose(0, 1))
        else:
            logits = torch.matmul((g_word * x + (1 - g_word) * r_sub), self.classifier.transpose(0, 1))

        if torch.any(torch.isnan(logits)):
            raise ValueError
        if self.na_att:
            return F.cross_entropy(logits[labels>=0], labels[labels>=0]) + F.cross_entropy(att_logits[labels>=0], labels[labels>=0]), logits
        else:
            return F.cross_entropy(logits[labels>=0], labels[labels>=0]), logits

    def normalize(self, x, scale:float=0.):
        return  x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-9 + scale)

    def update_history(self, x, labels):
        with torch.no_grad():
            for i in range(1, self.output_dim):
                if torch.sum((labels==i).long()) > 0:
                    pos_features = x[labels == i].clone().detach().mean(0, keepdim=True)
                    self.history_input[i-1] = self.mu * self.history_input[i-1] + pos_features
                    self.history_count[i-1] += 1
        if torch.any(torch.isnan(self.history_input)):
            raise ValueError
        return

    def multi_head_call(self, func, x, nheads, **kwargs):
        x_list = torch.split(x, self.head_dim, dim=-1)
        y_list = [func(item, **kwargs) for item in x_list]
        assert len(x_list) == nheads
        assert len(y_list) == nheads
        return torch.cat(y_list, dim=-1)

    def generate_substitute(self, r_ctx):
        w_logits = self.ctx_q(r_ctx)
        w = torch.softmax(w_logits, dim=-1)
        if self.mu < 1:
            v = self.history_input * (1 - self.mu) / (1 - torch.pow(self.mu, torch.clamp(self.history_count.unsqueeze(1), min=1)))
        else:
            v = self.history_input / self.history_count.unsqueeze(1)
        if self.na_att:
            substitute = torch.matmul(w[..., 1:], v)
        else:
            substitute = torch.matmul(w, v)
        return substitute, w_logits

    def reg(self,):
        return torch.mean(torch.matmul(self.contextual.weight.transpose(0, 1), self.word.weight)**2)


class SurrogateDistillClassifier(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, mu:float, lam:float, att_loss:bool=True, nheads:int=2, **kwargs):
        super().__init__()
        self.nheads = nheads

        self.classifier = nn.Parameter(torch.Tensor(output_dim, input_dim), requires_grad=True)
        stdv = 1. / math.sqrt(self.classifier.size(1))
        self.classifier.data.uniform_(-stdv, stdv)

        self.ctx_q = nn.Linear(input_dim, output_dim)
        self.word_g = nn.Linear(input_dim, 1)

        self.to_ctx_q = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

        self.output_dim = output_dim
        self.head_dim = input_dim // nheads
        self.mu = mu
        self.tau = 16 / self.nheads
        self.lam = lam
        self.use_cos = False
        self.att_loss = att_loss

        self.register_buffer("history_input", torch.FloatTensor(output_dim-1, input_dim).zero_())
        self.register_buffer("history_count", torch.LongTensor(output_dim-1).zero_())

    def p_loss(self, logits, plabels):
        y_na = (torch.sum(plabels, dim=-1, keepdim=True) < 0.5).float()
        y = torch.cat((y_na, plabels), dim=-1)
        y = y / torch.sum(y, dim=-1, keepdim=True)
        if torch.any(torch.isnan(y)):
            raise ValueError
        return -torch.mean(y * torch.log_softmax(logits, dim=-1))


    def forward(self, x,  labels, x_ctx=None, training=True, no_update_history:bool=False, no_logits_loss:bool=False):
        if self.use_cos:
            normed_w = self.multi_head_call(self.normalize, self.classifier, nheads=self.nheads, scale=self.lam)
        if training and not no_update_history:
            self.update_history(x, labels)
        r_sub, ctx_logits, att_logits = self.generate_substitute(x, x_ctx, training=training)
        g_word = torch.sigmoid(self.word_g(x)) * 0.5
        # print(g_word[:, 13, :], torch.softmax(att_logits, dim=-1)[:, 13, :])
        if self.use_cos:
            normed_x = self.multi_head_call(self.normalize, (g_word * x + (1 - g_word) * r_sub), nheads=self.nheads)
            logits = torch.matmul(normed_x * self.tau, normed_w.transpose(0, 1))
        else:
            logits = torch.matmul((g_word * x + (1 - g_word) * r_sub), self.classifier.transpose(0, 1))

        if torch.any(torch.isnan(logits)):
            raise ValueError
        if self.att_loss:
            if no_logits_loss:
                if training:
                    return F.cross_entropy(ctx_logits[labels>=0], labels[labels>=0]) + \
                        self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
                else:
                    return 0, logits
            else:
                if training:
                    return F.cross_entropy(logits[labels>=0], labels[labels>=0]) + \
                        F.cross_entropy(ctx_logits[labels>=0], labels[labels>=0]) + \
                        self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
                else:
                    return F.cross_entropy(logits[labels>=0], labels[labels>=0]), logits
        else:
            if no_logits_loss:
                if training:
                    return self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
                else:
                    return 0, logits
            else:
                if training:
                    return F.cross_entropy(logits[labels>=0], labels[labels>=0]) + \
                        self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
                else:
                    return F.cross_entropy(logits[labels>=0], labels[labels>=0]), logits

    def normalize(self, x, scale:float=0.):
        return  x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-9 + scale)

    def update_history(self, x, labels):
        with torch.no_grad():
            for i in range(1, self.output_dim):
                if torch.sum((labels==i).long()) > 0:
                    pos_features = x[labels == i].clone().detach().mean(0, keepdim=True)
                    self.history_input[i-1] = self.mu * self.history_input[i-1] + pos_features
                    self.history_count[i-1] += 1
        if torch.any(torch.isnan(self.history_input)):
            raise ValueError
        return

    def multi_head_call(self, func, x, nheads, **kwargs):
        x_list = torch.split(x, self.head_dim, dim=-1)
        y_list = [func(item, **kwargs) for item in x_list]
        assert len(x_list) == nheads
        assert len(y_list) == nheads
        return torch.cat(y_list, dim=-1)

    def generate_substitute(self, r, r_ctx, training=True):
        if training:
            ctx_logits = self.ctx_q(r_ctx)
        else:
            ctx_logits = None
        w_logits = self.to_ctx_q(r)
        if self.mu < 1:
            v = self.history_input * (1 - self.mu) / (1 - torch.pow(self.mu, torch.clamp(self.history_count.unsqueeze(1), min=1)))
        else:
            v = self.history_input / self.history_count.unsqueeze(1)
        if training and not self.att_loss:
            w = torch.softmax(ctx_logits, dim=-1)
        else:
            w = torch.softmax(w_logits, dim=-1)
        substitute = torch.matmul(w[..., 1:], v)
        return substitute, ctx_logits, w_logits

    def reg(self,):
        return torch.mean(torch.matmul(self.contextual.weight.transpose(0, 1), self.word.weight)**2)


class SurrogateDistillClassifierVar(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, mu:float, lam:float, att_loss:bool=True, nheads:int=2):
        super().__init__()
        self.nheads = nheads

        self.classifier = nn.Parameter(torch.Tensor(output_dim, input_dim), requires_grad=True)
        stdv = 1. / math.sqrt(self.classifier.size(1))
        self.classifier.data.uniform_(-stdv, stdv)

        self.none_tensor = nn.Parameter(torch.Tensor(1, input_dim), requires_grad=True)
        stdv = 1. / math.sqrt(self.none_tensor.size(1))
        self.none_tensor.data.uniform_(-stdv, stdv)

        self.ctx_q = nn.Linear(input_dim, output_dim)
        self.word_g = nn.Linear(input_dim, 1)

        self.to_ctx_q = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

        self.output_dim = output_dim
        self.head_dim = input_dim // nheads
        self.mu = mu
        self.tau = 16 / self.nheads
        self.lam = lam
        self.use_cos = False
        self.att_loss = att_loss

        self.register_buffer("history_input", torch.FloatTensor(output_dim-1, input_dim).zero_())
        self.register_buffer("history_count", torch.LongTensor(output_dim-1).zero_())

    def p_loss(self, logits, plabels):
        y_na = (torch.sum(plabels, dim=-1, keepdim=True) < 0.5).float()
        y = torch.cat((y_na, plabels), dim=-1)
        y = y / torch.sum(y, dim=-1, keepdim=True)
        if torch.any(torch.isnan(y)):
            raise ValueError
        return -torch.mean(y * torch.log_softmax(logits, dim=-1))


    def forward(self, x, x_ctx, labels, training=True, no_logits_loss:bool=False):
        if self.use_cos:
            normed_w = self.multi_head_call(self.normalize, self.classifier, nheads=self.nheads, scale=self.lam)
        if training:
            self.update_history(x, labels)
        r_sub, ctx_logits, att_logits = self.generate_substitute(x, x_ctx, training=training)

        # r_sub: B x L x C x d  ctx, att: B x L x C
        # word_g: B x L w = (word_g.unsqueeze(-1) * ctx).unsqueeze(-1),   (1 - w) * x.unsqueeze(-2) +  w * r_sub
        g_word = torch.sigmoid(self.word_g(x))
        # if self.use_cos:
        #     normed_x = self.multi_head_call(self.normalize, (g_word * x + (1 - g_word) * r_sub), nheads=self.nheads)
        #     logits = torch.matmul(normed_x * self.tau, normed_w.transpose(0, 1))
        # else:
        #     logits = torch.matmul((g_word * x + (1 - g_word) * r_sub), self.classifier.transpose(0, 1))
        # print(ctx_logits.size(), att_logits.size(), g_word.size())
        if training:
            w = (g_word * torch.sigmoid(ctx_logits)).unsqueeze(-1)
        else:
            w = (g_word * torch.sigmoid(att_logits)).unsqueeze(-1)

        r_pred = (1 - w) * x.unsqueeze(-2) +  w * r_sub
        logits = torch.sum(r_pred * self.classifier, dim=-1)


        if torch.any(torch.isnan(logits)):
            raise ValueError

        if no_logits_loss:
            return self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
        else:
            return F.cross_entropy(logits[labels>=0], labels[labels>=0]) + \
                self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits

    def normalize(self, x, scale:float=0.):
        return  x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-9 + scale)

    def update_history(self, x, labels):
        with torch.no_grad():
            for i in range(1, self.output_dim):
                if torch.sum((labels==i).long()) > 0:
                    pos_features = x[labels == i].clone().detach().mean(0, keepdim=True)
                    self.history_input[i-1] = self.mu * self.history_input[i-1] + pos_features
                    self.history_count[i-1] += 1
        if torch.any(torch.isnan(self.history_input)):
            raise ValueError
        return

    def multi_head_call(self, func, x, nheads, **kwargs):
        x_list = torch.split(x, self.head_dim, dim=-1)
        y_list = [func(item, **kwargs) for item in x_list]
        assert len(x_list) == nheads
        assert len(y_list) == nheads
        return torch.cat(y_list, dim=-1)

    def generate_substitute(self, r, r_ctx, training=True):
        ctx_logits = self.ctx_q(r_ctx)
        w_logits = self.to_ctx_q(r)
        if self.mu < 1:
            v = self.history_input * (1 - self.mu) / (1 - torch.pow(self.mu, torch.clamp(self.history_count.unsqueeze(1), min=1)))
        else:
            v = self.history_input / self.history_count.unsqueeze(1)
        # if training and not self.att_loss:
        #     w = torch.softmax(ctx_logits, dim=-1)
        # else:
        #     w = torch.softmax(w_logits, dim=-1)
        # substitute = torch.matmul(w[..., 1:], v)
        return torch.cat((self.none_tensor, v), dim=0), ctx_logits, w_logits

    def reg(self,):
        return torch.mean(torch.matmul(self.contextual.weight.transpose(0, 1), self.word.weight)**2)


class MixGate(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, use_ctx:bool=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        stdv1 = 1. / math.sqrt(output_dim-1)
        self.type_freq_w = nn.Linear(output_dim, 1)
        self.total_freq_w = nn.Parameter(torch.scalar_tensor(-1), requires_grad=True)
        self.total_freq_w.data.uniform_(-1, 1)
        self.sim_qw = nn.Parameter(torch.FloatTensor(output_dim-1), requires_grad=True)
        self.sim_qw.data.uniform_(-stdv1, stdv1)
        self.sim_w = nn.Parameter(torch.FloatTensor(output_dim-1), requires_grad=True)
        self.sim_w.data.uniform_(-stdv1, stdv1)
        self.sim_b = nn.Parameter(torch.FloatTensor(output_dim-1), requires_grad=True)
        self.sim_b.data.uniform_(-stdv1, stdv1)
        self.sim_gw = nn.Linear(output_dim-1, 1)

        self.feat1_w = nn.Linear(input_dim, 1)
        self.feat2_w = nn.Linear(input_dim, 1)
        self.scale_factor = 1 # 10 if output_dim < 100 else 1# ace 10, maven 1
        self.ctx_w = nn.Linear(input_dim, 1) if use_ctx else None

        self.bm = nn.BatchNorm1d(num_features=5 if use_ctx else 4)
        self.feat_drop = nn.Dropout(0.5)
        self.inp_drop = nn.Dropout(0.2)

    def _apply_bm(self, x):
        return self.bm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, feat1, feat2, sim, freqs, total_freqs, ctx=None):
        sim, freqs, total_freqs = self.inp_drop(sim), self.inp_drop(freqs), self.inp_drop(total_freqs)

        sim_h = sim ** 2 * self.sim_qw + sim * self.sim_w + self.sim_b
        sim_val = self.sim_gw(sim_h)
        type_freq_val = self.type_freq_w(freqs)
        total_freq_val = self.total_freq_w * total_freqs
        feat_val = self.feat1_w(self.feat_drop(feat1)) + self.feat2_w(self.feat_drop(feat2))
        if ctx is not None and self.ctx_w is not None:
            ctx_val = self.ctx_w(self.feat_drop(ctx))
            vals = self._apply_bm(torch.cat((sim_val, type_freq_val, total_freq_val, feat_val, ctx_val), dim=-1))
            g_word = torch.sigmoid(torch.mean(vals, dim=-1, keepdim=True))
        else:
            vals = self._apply_bm(torch.cat((sim_val, type_freq_val, total_freq_val, feat_val), dim=-1))
            g_word = torch.sigmoid(torch.mean(vals, dim=-1, keepdim=True))
        return g_word



class SurrogateDistillClassifierLayer(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, mu:float, lam:float, att_loss:bool=True, nheads:int=2, fusion_layer:int=0, token_freq_tensor:Optional[torch.Tensor]=None, type_token_tensor:Optional[Dict]=None):
        super().__init__()
        self.nheads = nheads


        # a(x - s)^2 + b = a x^2 - 2 a s x + a s^2 + b = a x^2 - c x  + d
        self.classifier = nn.Parameter(torch.Tensor(output_dim, input_dim), requires_grad=True)
        stdv = 1. / math.sqrt(self.classifier.size(1))
        self.classifier.data.uniform_(-stdv, stdv)

        if token_freq_tensor is not None:
            self.fgate = MixGate(input_dim, output_dim, use_ctx=True)
            self.sgate = MixGate(input_dim, output_dim, use_ctx=True)
            # self.sgate = MixGate(input_dim, output_dim)

            # stdv1 = 1. / math.sqrt(output_dim-1)
            # self.type_freq_w = nn.Linear(output_dim, 1)
            # self.total_freq_w = nn.Parameter(torch.scalar_tensor(-1), requires_grad=True)
            # self.total_freq_w.data.uniform_(-1, 0)
            # self.sim_qw = nn.Parameter(torch.FloatTensor(output_dim-1), requires_grad=True)
            # self.sim_qw.data.uniform_(-stdv1, 0)
            # self.sim_w = nn.Parameter(torch.FloatTensor(output_dim-1), requires_grad=True)
            # self.sim_w.data.uniform_(-stdv1, 0)
            # self.sim_b = nn.Parameter(torch.FloatTensor(output_dim-1), requires_grad=True)
            # self.sim_b.data.uniform_(-stdv1, 0)
            # self.sim_gw = nn.Linear(output_dim-1, 1)


        # self.ctx_q = nn.Linear(input_dim, output_dim)
        # self.word_g = nn.Linear(input_dim, 1)

        # self.to_ctx_q = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.GELU(),
        #     nn.Linear(input_dim, output_dim)
        # )
        self.to_ctx_q = nn.Linear(input_dim, output_dim)

        self.output_dim = output_dim
        self.head_dim = input_dim // nheads
        self.mu = mu
        self.tau = 16 / self.nheads
        self.lam = lam
        self.use_cos = False
        self.att_loss = att_loss
        self.fusion_layer = fusion_layer

        if token_freq_tensor is not None:
            scale_factor = 0.1
            self.register_buffer("token_frequency", torch.tanh(scale_factor * token_freq_tensor))
            self.register_buffer("token_total_frequency", torch.tanh(scale_factor * torch.sum(token_freq_tensor, dim=-1, keepdim=True)))
        else:
            self.token_frequency = self.token_total_frequency = None

        if type_token_tensor is not None:
            self.register_buffer("embedding_ids", type_token_tensor["embedding_ids"])
            self.register_buffer("type_weight", type_token_tensor["type_weight"])
            assert self.fusion_layer == 0
            self.use_type_token = True
        else:
            self.register_buffer("history_input", torch.FloatTensor(output_dim-1, input_dim).zero_())
            self.register_buffer("history_count", torch.LongTensor(output_dim-1).zero_())
            self.use_type_token = False

    def p_loss(self, logits, plabels):
        y_na = (torch.sum(plabels, dim=-1, keepdim=True) < 0.5).float()
        y = torch.cat((y_na, plabels), dim=-1)
        y = y / torch.sum(y, dim=-1, keepdim=True)
        if torch.any(torch.isnan(y)):
            raise ValueError
        return -torch.mean(y * torch.log_softmax(logits, dim=-1))

    def pre_forward(self, batch, bert_model):
        return surrogate_pre_forward(bert_model, batch, self.fusion_layer)


    def forward(self, batch, bert_model, x_ctx, labels, training=True, no_update_history:bool=False, no_logits_loss:bool=False, ind=None):

        x = surrogate_pre_forward(bert_model, batch, self.fusion_layer)

        if self.token_frequency is not None:
            freqs = F.embedding(batch['input_ids'], self.token_frequency)
            total_freqs = F.embedding(batch['input_ids'], self.token_total_frequency)
        else:
            freqs = total_freqs = None


        if training and not no_update_history and not self.use_type_token:
            self.update_history(x, labels)
        # r_sub, ctx_logits, att_logits = self.generate_substitute(x, x_ctx, training=training)
        # g_word = torch.sigmoid(self.word_g(x))
        r_sub, g_word, att_logits = self.generate_substitute2(x, r_ctx=batch["context_features"],training=training, freqs=freqs, total_freqs=total_freqs, bert_model=bert_model)
        ctx_logits = att_logits
        # g_word = (1, 0)

        # if self.lam > 0:
        #     g_word = g_word * self.lam
        # else:
        #     g_word = (1 - g_word) * self.lam + 1

        if ind:
            _att = torch.softmax(att_logits[..., 1:], dim=-1)[:, ind, :].topk(k=5)
            if isinstance(g_word, tuple):
                for _g_word in g_word:
                    print(_g_word[:, ind, :], _att.values, _att.indices+1)
            else:
                print(g_word[:, ind, :], _att.values, _att.indices+1)
        h = surrogate_post_forward(bert_model, batch, self.fusion_layer, x, r_sub, g_word)
        logits = torch.matmul(h, self.classifier.transpose(0, 1))

        if torch.any(torch.isnan(logits)):
            raise ValueError
        if self.att_loss:
            if no_logits_loss:
                return self.lam * F.cross_entropy(ctx_logits[labels>=0], labels[labels>=0]) + \
                    self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
            else:
                return F.cross_entropy(logits[labels>=0], labels[labels>=0]) + \
                    self.lam * F.cross_entropy(ctx_logits[labels>=0], labels[labels>=0]) + \
                    self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
        else:
            if no_logits_loss:
                return self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits
            else:
                return F.cross_entropy(logits[labels>=0], labels[labels>=0]) + \
                    self.lam * torch.mean((ctx_logits[labels>=0] - att_logits[labels>=0]) ** 2), logits

    def normalize(self, x, scale:float=0.):
        return  x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-9 + scale)

    def update_history(self, x, labels):
        with torch.no_grad():
            for i in range(1, self.output_dim):
                if torch.sum((labels==i).long()) > 0:
                    pos_features = x[labels == i].clone().detach().mean(0, keepdim=True)
                    self.history_input[i-1] = self.mu * self.history_input[i-1] + pos_features
                    self.history_count[i-1] += 1
        if torch.any(torch.isnan(self.history_input)):
            raise ValueError
        return

    def multi_head_call(self, func, x, nheads, **kwargs):
        x_list = torch.split(x, self.head_dim, dim=-1)
        y_list = [func(item, **kwargs) for item in x_list]
        assert len(x_list) == nheads
        assert len(y_list) == nheads
        return torch.cat(y_list, dim=-1)

    def generate_substitute(self, r, r_ctx, training=True):
        ctx_logits = self.ctx_q(r_ctx)
        w_logits = self.to_ctx_q(r)
        if self.mu < 1:
            v = self.history_input * (1 - self.mu) / (1 - torch.pow(self.mu, torch.clamp(self.history_count.unsqueeze(1), min=1)))
        else:
            v = self.history_input / self.history_count.unsqueeze(1)
        # w = torch.softmax(ctx_logits, dim=-1)
        w = torch.softmax(ctx_logits, dim=-1)
        # if training and not self.att_loss:
        #     w = torch.softmax(ctx_logits, dim=-1)
        # else:
        #     w = torch.softmax(w_logits, dim=-1)
        substitute = torch.matmul(w[..., 1:], v)
        return substitute, ctx_logits, w_logits

    def generate_substitute2(self, r, r_ctx=None, freqs=None, total_freqs=None, training=True, bert_model=None):
        if r_ctx is not None:
            w_logits = self.to_ctx_q(r)
        else:
            w_logits = self.to_ctx_q(r_ctx)

        if self.use_type_token:
            fetched_embeddings = bert_model.embeddings.word_embeddings(self.embedding_ids)
            v = torch.matmul(self.type_weight, fetched_embeddings)
        else:
            if self.mu < 1:
                v = self.history_input * (1 - self.mu) / (1 - torch.pow(self.mu, torch.clamp(self.history_count.unsqueeze(1), min=1)))
            else:
                v = self.history_input / self.history_count.unsqueeze(1)
        # w = torch.softmax(ctx_logits, dim=-1)
        # if training and not self.att_loss:
        #     w = torch.softmax(ctx_logits, dim=-1)
        # else:
        #     w = torch.softmax(w_logits, dim=-1)

        if freqs is not None:
            sim = torch.sum(self.normalize(r.unsqueeze(-2)) * self.normalize(v), dim=-1, keepdim=False)
            # sim_val = torch.max(sim, dim=-1, keepdim=True).values
            # sim_val = self.sim_gw(sim ** 2 * self.sim_qw + sim * self.sim_w + self.sim_b)
            # freq_val = self.type_freq_w(freqs) + self.total_freq_w * total_freqs
            substitute = torch.matmul(torch.softmax(w_logits[..., 1:], dim=-1), v)

            fg = self.fgate.forward(r, substitute, sim, freqs, total_freqs, r_ctx)
            sg = self.sgate.forward(r, substitute, sim, freqs, total_freqs, r_ctx)

            g_word = (fg, sg)
            # g_word = fg
        else:
            w = torch.softmax(w_logits, dim=-1)
            substitute = torch.matmul(w[..., 1:], v)
            sim = torch.sum(self.normalize(r) * self.normalize(substitute), dim=-1, keepdim=True)
            sim_val = torch.max(sim, dim=-1, keepdim=True).values * -4
            g_word = torch.sigmoid(sim_val)
        return substitute, g_word, w_logits

    def reg(self,):
        return torch.mean(torch.matmul(self.contextual.weight.transpose(0, 1), self.word.weight)**2)
