from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import *
from .utils import Filters, MomentumClassifier, SimpleCRFHead, SurrogateClassifier#, SurrogateDistillClassifier
from .utils import SurrogateDistillClassifierLayer as SurrogateDistillClassifier
import math

class Config(object):
    def __init__(self, kwargs):
        super().__init__()
        self.valid_keys = set()
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.valid_keys.add(k)
    def to_dict(self,):
        return {k: getattr(self, k) for k in self.valid_keys}

class SeqCls(nn.Module):
    def __init__(self, nclass:int, model_name:str, use_crf=False, task_of_label:str='trigger',
        focal:bool=False, alpha:float=-1, gamma:float=-1,
        featurewise:bool=False, kernel_size:int=64, n_dilations:int=3, lam:float=1e-3,
        topicwise:bool=False, topic_emb_path:str="", 
        momentum:bool=False, n_momentum_heads:int=2, momentum_norm_factor:float=1./32, momentum_weight:float=1.5, mu:float=0.9998, 
        tau_norm:bool=False, tau:float=1., tau_norm_bias:bool=False,
        crt:bool=False, class_balance_tensor:Optional[torch.FloatTensor]=None, 
        lws:bool=False, lws_bias:bool=False,
        ncm:bool=False, ncm_th:float=0.,
        surrogate:bool=False, surrogate_distill:bool=False, surrogate_mu:float=0.9, surrogate_lam:float=1., surrogate_na:bool=False, surrogate_att_loss:bool=True, surrogate_fusion_layer:int=0,
        surrogate_lws:bool=False, token_freq_tensor:Optional[torch.FloatTensor]=None, type_token_tensor:Optional[Dict]=None, **kwargs):
        super().__init__()
        self.pretrained_lm = transformers.AutoModel.from_pretrained(model_name)
        d_model = 768 if "bert-base" in model_name else getattr(self.pretrained_lm.config, 'd_model', 1024)
        self.d_model = d_model
        self.use_crf = use_crf
        if self.use_crf:
            self.crf_head = SimpleCRFHead(nstate=nclass)
        self.event_cls = nn.Linear(d_model, nclass)
        self.task_of_label = task_of_label
        self.nclass = nclass
        self.alpha = alpha
        self.gamma = gamma
        self.focal = focal
        self.featurewise = featurewise
        self.topicwise = topicwise
        self.momentum = momentum
        self.tau_norm = tau_norm
        self.crt = crt
        self.lws = lws
        self.surrogate = surrogate
        self.surrogate_distill = surrogate_distill
        self.surrogate_lws = surrogate_lws
        self.ncm = ncm
        if sum([1 if t else 0 for t in [focal, featurewise, topicwise, momentum, tau_norm, crt, lws, surrogate, surrogate_distill]]) > 1: raise ValueError("Cannot build model for two methods")
        if self.featurewise:
            self.feature_filters = Filters(d_model, nclass, kernel_size, dilations=[2**t for t in range(n_dilations)], lam=lam)
            self.feature_factor = - math.log(self.feature_filters.n_heads)
        if self.topicwise:
            self.topic_emb = nn.Embedding.from_pretrained(embeddings=torch.load(topic_emb_path), freeze=True)
        if self.momentum:
            self.momentum_config = Config({
                "nheads": n_momentum_heads,
                "alpha": momentum_weight,
                "gamma": momentum_norm_factor,
                "tau": 16.0 / n_momentum_heads,
                "mu": mu
            })
            self.momentum_cls = MomentumClassifier(input_dim=d_model, output_dim=self.nclass, **self.momentum_config.to_dict())
        if self.tau_norm:
            self.tau_norm_config = Config({
                "tau": tau,
                "tau_norm_bias": tau_norm_bias
            })
        if self.crt:
            self.class_weight = nn.Parameter(class_balance_tensor, requires_grad=False)
            self.pretrained_lm.requires_grad_(False)
        if self.lws:
            self.lws_weight = nn.Parameter(torch.ones(self.nclass), requires_grad=True)
            self.class_weight = nn.Parameter(class_balance_tensor, requires_grad=False)
            self.pretrained_lm.requires_grad_(False)
            self.event_cls.requires_grad_(False)
            self.lws_config = Config({
                "lws_bias": lws_bias
            })
        if self.surrogate:
            self.surrogate_config = Config({
                "input_dim": d_model,
                "output_dim": nclass,
                "mu": surrogate_mu,
                "lam": surrogate_lam,
                "na_att": surrogate_na,
                "token_freq_tensor": token_freq_tensor,
                "type_token_tensor": type_token_tensor
            })
            self.surrogate_cls = SurrogateClassifier(**self.surrogate_config.to_dict())
            if self.surrogate_lws:
                self.lws_weight = nn.Parameter(torch.ones(self.nclass), requires_grad=True)
                self.class_weight = nn.Parameter(class_balance_tensor, requires_grad=False)
                self.pretrained_lm.requires_grad_(False)
                self.surrogate_cls.requires_grad_(False)
        if self.surrogate_distill:
            self.surrogate_distill_config = Config({
                "input_dim": d_model,
                "output_dim": nclass,
                "mu": surrogate_mu,
                "lam": surrogate_lam,
                "att_loss": surrogate_att_loss,
                "fusion_layer": surrogate_fusion_layer,
                "token_freq_tensor": token_freq_tensor,
                "type_token_tensor": type_token_tensor
            })
            self.surrogate_distill_cls = SurrogateDistillClassifier(**self.surrogate_distill_config.to_dict())
            self.pretrained_lm.requires_grad_(False)
            if self.surrogate_lws:
                self.lws_weight = nn.Parameter(torch.ones(self.nclass), requires_grad=True)
                self.class_weight = nn.Parameter(class_balance_tensor, requires_grad=False)
                self.pretrained_lm.requires_grad_(False)
                self.surrogate_distill_cls.requires_grad_(False)
        if self.ncm:
            self.ncm_config = Config({"threshold": ncm_th})

    def compute_loss(self, logits, labels, crit=None):
        mask = labels >= 0
        if crit is None: crit = self.focal_loss if self.focal else F.cross_entropy
        return crit(input=logits[mask], target=labels[mask], weight=self.class_weight if (self.crt or self.lws or self.surrogate_lws) else None)

    def focal_loss(self, input, target, alpha:Optional[float]=None, gamma:Optional[float]=None, *args, **kwargs):
        alpha = alpha if alpha else self.alpha
        gamma = gamma if gamma else self.gamma
        if alpha > 0 and alpha < 1:
            weight = torch.tensor(
                [alpha] + [1- alpha] * (self.nclass-1),
                dtype=torch.float,
                device=input.device,
                requires_grad=False)
        else:
            weight = None
        loss = F.cross_entropy(input=input, target=target, weight=weight, reduction='none')
        if gamma > 0:
            prob = torch.softmax(input, dim=-1)
            pweight = (1 - prob[torch.arange(prob.size(0), device=prob.device), target]) ** gamma
            loss = pweight * loss
        if weight is None:
            loss = loss.mean()
        else:
            cweight = torch.sum(weight[target] * pweight)
            loss = loss.sum() / (cweight + 1e-6)
        return loss

    def backdoor_loss(self, inputs, target):
        probs = [torch.log_softmax(i, dim=-1) for i in inputs]
        mask = target >= 0
        masked_target = target[mask]
        masked_probs = [t[mask] for t in probs]
        selected_probs = torch.cat([torch.gather(t, dim=-1, index=masked_target.unsqueeze(1)) for t in masked_probs], dim=-1) + self.feature_factor

        if self.focal:
            weights = sum([torch.softmax(i, dim=-1) for i in inputs]) / len(inputs)
            weights = weights[mask]
            selected_weights = torch.gather(weights, dim=-1, index=masked_target.unsqueeze(1)).squeeze(1)
            selected_weights = (1 - selected_weights) ** self.gamma
            selected_weights *= ((masked_target == 0).float() * self.alpha + (masked_target > 0).float() * (1-self.alpha))
            loss = -torch.sum(torch.logsumexp(selected_probs, dim=-1) * selected_weights) / torch.clamp(torch.sum(selected_weights), 1e-6)
        else:
            loss = -torch.mean(torch.logsumexp(selected_probs, dim=-1))
        if self.featurewise:
            reg = torch.sum((torch.matmul(self.feature_filters.projection.weight, self.feature_filters.projection.weight.transpose(0, 1)) - self.feature_filters.eye)**2)
            return loss + self.feature_filters.lam * reg
        else:
            return loss

    def topic_foward(self, hidden):
        pass
        
    def soft_binary(self, logits, labels):
        return torch.mean(- labels * torch.log_softmax(logits, dim=-1))

    def forward(self, batch, predict=False, return_output=False, **kwargs):
        if not self.surrogate_distill:
            encoded = self.pretrained_lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        labels = batch[f"{self.task_of_label}_labels"]

        if self.featurewise:
            outputs = self.feature_filters.forward(encoded.last_hidden_state)
            loss = self.backdoor_loss(outputs, labels)
            outputs = sum([torch.softmax(output, dim=-1) for output in outputs])
        elif self.momentum:
            outputs = self.momentum_cls.forward(encoded.last_hidden_state, labels, training=not predict)
            if predict:
                logits, outputs = outputs
            else:
                logits = outputs
            loss = self.compute_loss(logits, labels)
        elif self.tau_norm and predict:
            weights = self.event_cls.weight.clone()
            weights = weights / torch.norm(weights, dim=1, p=2, keepdim=True) ** self.tau_norm_config.tau
            outputs = torch.matmul(encoded.last_hidden_state, weights.transpose(0, 1))
            if self.tau_norm_config.tau_norm_bias:
                bias = self.event_cls.bias.clone() / torch.norm(weights, dim=1, p=2) ** self.tau_norm_config.tau
                outputs = outputs + bias
            loss = self.compute_loss(outputs, labels)
        elif self.lws:
            if self.lws_config.lws_bias:
                outputs = self.event_cls(encoded.last_hidden_state) * self.lws_weight
            else:
                weights = self.event_cls.weight * self.lws_weight.unsqueeze(1)
                outputs = torch.matmul(encoded.last_hidden_state, weights.transpose(0, 1))
            loss = self.compute_loss(outputs, labels)
        elif self.surrogate:
            loss, outputs = self.surrogate_cls.forward(x=encoded.last_hidden_state, labels=labels, x_ctx=batch["context_features"],training=not predict)
            if self.surrogate_lws:
                loss = self.compute_loss(outputs * self.lws_weight, labels)
        elif self.surrogate_distill:
            # loss, outputs = self.surrogate_distill_cls.forward(x=encoded.last_hidden_state, labels=labels, x_ctx=None if predict else batch["context_features"],training=not predict, no_update_history=self.surrogate_lws, no_logits_loss=self.use_crf)
            loss, outputs = self.surrogate_distill_cls.forward(
                batch=batch,
                bert_model=self.pretrained_lm,
                labels=labels, 
                x_ctx=None,
                training=not predict, 
                no_update_history=self.surrogate_lws, 
                no_logits_loss=self.use_crf,
                **kwargs)
            
            if self.surrogate_lws:
                outputs = outputs * self.lws_weight
                loss = self.compute_loss(outputs, labels)
            if self.use_crf:
                path = labels.clone()
                path[path < 0] = 0
                loss = loss + torch.mean(self.crf_head(outputs, path, batch["attention_mask"]>0))
        elif self.ncm:
            normed_x = encoded.last_hidden_state / torch.clamp(torch.norm(encoded.last_hidden_state, p=2, dim=-1, keepdim=True), min=1e-9)
            normed_y = self.history_input / torch.clamp(torch.norm(self.history_input, p=2, dim=-1, keepdim=True), min=1e-9)
            score = torch.matmul(normed_x, normed_y.transpose(0, 1))
            na_score = torch.ones(score.size(0), score.size(1), 1, device=score.device) * self.ncm_config.threshold
            outputs = torch.cat((na_score, score), dim=-1)
            loss = 0.
        else:
            outputs = self.event_cls(encoded.last_hidden_state)
            path = labels.clone()
            path[path < 0] = 0
            scores = self.crf_head(outputs, path, batch["attention_mask"]>0) if self.use_crf else self.compute_loss(outputs, labels)
            loss = torch.mean(scores)
        if self.use_crf and predict:
            preds, _ = self.crf_head.prediction(outputs)
        else:
            preds = torch.argmax(outputs, dim=-1)
        preds[labels < 0] = labels[labels < 0]

        if return_output:
            return {
                "loss": loss,
                "prediction": preds.long().detach().cpu(),
                "label": labels.long().detach().cpu(),
                "output": outputs.detach().cpu(),
                }
        else:
            return {
                "loss": loss,
                "prediction": preds.long().detach().cpu(),
                "label": labels.long().detach().cpu()
                }
    
    
    def update_history(self, batch):
        if getattr(self, "history_input", None) is None: self.history_input = torch.FloatTensor(self.nclass-1, self.d_model).zero_().to(batch["input_ids"].device)
        if getattr(self, "history_count", None) is None: self.history_count = torch.LongTensor(self.nclass-1).zero_().to(batch["input_ids"].device)

        if self.surrogate_distill:
            encoded = self.surrogate_distill_cls.pre_forward(batch, self.pretrained_lm)
        else:
            encoded = self.pretrained_lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        labels = batch[f"{self.task_of_label}_labels"]
        with torch.no_grad():
            for i in range(1, self.nclass):
                if torch.sum((labels==i).long()) > 0:
                    pos_features = encoded[labels == i].clone().detach().mean(0, keepdim=True)
                    self.history_input[i-1] = (self.history_input[i-1] * self.history_count[i-1] + pos_features) / (self.history_count[i-1] + 1)
                    self.history_count[i-1] += 1
        if torch.any(torch.isnan(self.history_input)):
            raise ValueError
        return