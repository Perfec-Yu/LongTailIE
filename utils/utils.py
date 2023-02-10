from copy import deepcopy
from .data import IDataset, _to_instance
import json
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import os
import re
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import PreTrainedTokenizerFast, BatchEncoding, AutoTokenizer
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from typing import *


COUNTS = {
    "ace": [293188, 1344, 637, 492, 235, 162, 161, 187, 108, 82, 87, 83, 106, 99, 65, 67, 85, 67, 49, 54, 49, 54, 49, 39, 31, 43, 12, 16, 9, 10, 22, 6, 1, 1],
    "maven": [924893, 847, 186, 3632, 748, 3412, 1347, 426, 164, 812, 2448, 1038, 1320, 1017, 365, 789, 684, 284, 253, 2946, 1698, 1549, 116, 3190, 933, 467, 805, 381, 512, 1169, 683, 3404, 287, 464, 1087, 22, 1427, 1012, 221, 630, 632, 942, 557, 879, 366, 1080, 1459, 107, 104, 239, 822, 626, 241, 530, 698, 357, 108, 142, 877, 347, 966, 1471, 136, 296, 155, 803, 199, 106, 592, 638, 861, 482, 244, 389, 115, 173, 128, 730, 440, 493, 55, 329, 1748, 85, 2571, 99, 145, 741, 4, 1007, 215, 541, 48, 88, 450, 180, 871, 830, 285, 352, 180, 69, 226, 218, 105, 311, 293, 41, 236, 103, 183, 21, 6, 95, 334, 184, 135, 83, 179, 501, 243, 15, 166, 138, 102, 117, 858, 291, 43, 1115, 512, 362, 36, 379, 120, 355, 83, 372, 148, 138, 51, 235, 82, 65, 34, 201, 51, 64, 101, 17, 138, 16, 41, 116, 50, 56, 29, 7, 301, 20, 13, 146, 72, 78, 87, 84, 21, 15, 60],
    "fewnerd": [3027069, 39507, 18395, 17252, 7602, 92776, 60290, 43850, 26170, 9439, 8950, 1857, 18391, 21362, 136777, 26450, 32035, 8386, 19272, 37289, 8318, 5490, 5430, 9070, 15325, 13426, 6121, 13810, 11952, 5632, 8202, 12326, 21542, 8433, 7874, 30928, 11646, 8670, 6996, 13093, 4731, 8533, 31339, 3839, 5149, 15543, 20064, 9127, 5068, 7011, 10467, 2425, 11459, 5401, 14274, 4707, 2811, 12053, 9051, 2434, 7350, 5881, 5865, 5024, 5497, 5219, 1264]
}

def get_type_token_tensor(root:str):
    return torch.load(os.path.join(root, "weighted_type_tokens.th"))

def get_freq_tensor(root:str):
    return torch.load(os.path.join(root, "token_freq_mat.th"))

def get_maven_candidates(root:str, lm_name:str):
    with open(os.path.join(root, '_test.jsonl')) as fp:
        _test = [json.loads(t) for t in fp]
    with open(os.path.join(root, 'test.jsonl')) as fp:
        test = [json.loads(t) for t in fp]
    
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    candidates = []
    for i, t in enumerate(_test):
        candidates.append(set())
        candidate_dict ={}
        encoding:BatchEncoding = tokenizer(t['sentence'])
        for e in t['event_mentions']:
            start = encoding.char_to_token(e['trigger']['start'])
            end = encoding.char_to_token(e['trigger']['end']-1)+1
            candidate_dict[(start, end)] = e['event_type']
        for e in test[i]['event_mentions']:
            start = encoding.char_to_token(e['trigger']['start'])
            end = encoding.char_to_token(e['trigger']['end']-1)+1
            candidate_dict[(start, end)] = e['event_type']
        for (s, e), t in candidate_dict.items():
            candidates[-1].add((s, e, t))
    
    return candidates

def get_weight_tensor(dataset:str, leave_na:bool=False):
    counts = COUNTS[dataset.lower()].copy()
    
    K = len(counts) - 1
    if leave_na:
        counts = torch.FloatTensor(counts)
        N = torch.sum(counts)
        n0 = counts[0]
        M = N / (torch.sum(1. / counts[1:]) + K / (N - n0))
        w = torch.zeros_like(counts)
        w[0] = K * M / (N - n0)
        w[1:] = M / counts[1:]
    else:
        w = 1. / torch.FloatTensor(counts)
    w = w / torch.sum(w) * (K + 1)
    return w

def preprocess_func_token(input_file):
    detokenizer = TreebankWordDetokenizer()
    with open(input_file, "rt") as fp:
        data = [json.loads(t) for t in fp]
    data_sentences = [(str(i), detokenizer.detokenize(t['tokens'])) for i, t in enumerate(data)]
    sentences = set()
    data_sentences_no_replica = []
    for d in data_sentences:
        if d[1] not in sentences:
            sentences.add(d[1])
            data_sentences_no_replica.append(d)
    return data_sentences_no_replica

def create_optimizer_and_scheduler(model:torch.nn.Module, learning_rate:float, weight_decay:float, warmup_step:int, train_step:int, adam_beta1:float=0.9, adam_beta2:float=0.999, adam_epsilon:float=1e-8):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "lr": learning_rate,
        "betas": (adam_beta1, adam_beta2),
        "eps": adam_epsilon,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    scheduler = get_scheduler(
                "linear",
                optimizer,
                num_warmup_steps=warmup_step,
                num_training_steps=train_step,
            )
    return optimizer, scheduler

def get_dataset_and_loader(path, label2id:Dict[str, int], method='token', model_name='bert-large-cased', batch_size=8):
    try:
        with open(path, "rt") as f:
            data = json.load(f)
    except Exception as e:
        with open(path, "rt") as f:
            data = [json.loads(line) for line in f]
    if 'annotations' in data[0]:
        for example in data:
            for annotation in example['annotations']:
                if annotation[2].startswith("CND"):
                    annotation[2] = annotation[2][4:]
    instances = _to_instance(data)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = IDataset(
        instances=instances, 
        label2id=label2id,
        setting=method,
        tokenizer=tokenizer,
        max_length=128)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True)
    return dataset, dataloader


class Record(object):
    def __init__(self, percentage=False):
        super().__init__()
        self.value = 0.
        self.num = 0.
        self.percentage = percentage

    def __iadd__(self, val):
        self.value += val
        self.num += 1
        return self

    def reset(self):
        self.value = 0.
        self.num = 0.

    def __str__(self):
        if self.percentage:
            display = f"{self.value / max(1, self.num) * 100:.2f}%"
        else:
            display = f"{self.value / max(1, self.num):.4f}"
        return display

    @property
    def true_value(self,):
        return self.value / max(1, self.num)

    def __eq__(self, other):
        return self.true_value == (other if isinstance(other, float) else other.true_value)

    def __lt__(self, other):
        return self.true_value < (other if isinstance(other, float) else other.true_value)

    def __gt__(self, other):
        return self.true_value > (other if isinstance(other, float) else other.true_value)

    def __ge__(self, other):
        return self.true_value >= (other if isinstance(other, float) else other.true_value)

    def __le__(self, other):
        return self.true_value <= (other if isinstance(other, float) else other.true_value)

    def __ne__(self, other):
        return self.true_value != (other if isinstance(other, float) else other.true_value)

class F1Record(Record):
    def __init__(self, init=None):
        super().__init__()
        if init:
            self.value = np.array(init)
        else:
            self.value = np.zeros(3)
    def __iadd__(self, val:Union[Record, np.ndarray]):
        self.value += val
        return self
    def reset(self,):
        self.value = np.zeros(3)
    def __str__(self):
        denom = self.value[0] + self.value[1]
        return str(self.full_result)
    
    def round(self, v):
        return round(v * 1e4) / 1e4

    @property
    def true_value(self,):
        denom = self.value[0] + self.value[1]
        if denom == 0:
            return 0
        else:
            return self.round(self.value[2]*2 / denom)
    
    @property
    def full_result(self,):
        precision = self.value[2] / max(self.value[1], 1)
        recall = self.value[2] / max(self.value[0], 1)
        f1 = self.value[2] * 2/ max(self.value[0] + self.value[1], 1)
        return self.round(precision), self.round(recall), self.round(f1)
    
    @property
    def named_result(self,):
        precision, recall, f1 = self.full_result
        return {"precision": precision, "recall": recall, "f1":f1}


class MacroF1Record(Record):
    def __init__(self, prf):
        super().__init__()
        if isinstance(prf, dict):
            self.value = prf
    
    def __str__(self):
        return str(self.full_result)
    
    def round(self, v):
        return round(v * 1e4) / 1e4

    @property
    def true_value(self,):
        return self.round(self.value['f1'])
    
    @property
    def full_result(self,):
        return self.round(self.value['precision']), self.round(self.value['recall']), self.round(self.value['f1'])
    
    @property
    def named_result(self,):
        return self.value


class MixF1Record(Record):
    def __init__(self, prf1, prf2, name1, name2):
        super().__init__()
        self.value = prf1
        self.prop = prf2
        self.name1 = name1
        self.name2 = name2
    
    def __str__(self):
        # return str(self.full_result)
        r = self.full_result
        p1, r1, f1 = r[0]
        p2, r2, f2 = r[1]
        s = f"{self.name1}[P: {p1* 100:.2f}, R: {r1* 100:.2f}, F: {f1* 100:.2f}]|{self.name2}[P: {p2* 100:.2f}, R: {r2* 100:.2f}, F: {f2* 100:.2f}]"
        return s

    @property
    def true_value(self,):
        return self.value.true_value
    @property
    def full_result(self,):
        return [self.value.full_result, self.prop.full_result]
    @property
    def named_result(self,):
        out = {f'{self.name1}_{k}': v for k, v in self.value.named_result.items()}
        out.update({f'{self.name2}_{k}': v for k, v in self.prop.named_result.items()})
        return out


class F1MetricTag(object):

    NAL_match = re.compile(r'[^A-Z,a-z]')
    BIO_match = re.compile(r'(?P<start>\d+)B-(?P<label>[a-z]+)\s(?:(?P<end>\d+)I-(?P=label)\s)*')
    IO_match = re.compile(r'(?P<start>\d+)I-(?P<label>[a-z]+)\s(?:(?P<end>\d+)I-(?P=label)\s)*')

    def __init__(self, pad_value:int, ignore_labels:Optional[Union[int, List[int], Set[int]]], label2id:Dict[str, int], tokenizer:Optional[PreTrainedTokenizerFast]=None, fix_span:str='none', save_dir:Optional[str]=None, save_name:Optional[str]=None, save_output:bool=False, save_annotation:bool=False, return_annotation:bool=False, word_level:bool=False, candidates=None) -> None:
        if isinstance(ignore_labels, int):
            self.ignore_labels = {ignore_labels}
        elif ignore_labels is None:
            self.ignore_labels = set()
        else:
            self.ignore_labels = set(ignore_labels)
        self.pad_value = pad_value
        self.save_dir = save_dir
        self.save_name = save_name if save_name else "output.out"
        self.label2id = label2id
        self.fix_span = fix_span
        self.candidates = candidates
        self.id2nickname = {}
        self.nickname2label = {}
        self.id2tag = {}
        for label, id_ in label2id.items():
            uncased_label = self.NAL_match.sub('', label).lower()
            while uncased_label in self.nickname2label:
                uncased_label += 'a'
            self.nickname2label[uncased_label] = label
            self.id2nickname[id_] = uncased_label
        for id_, nickname in self.id2nickname.items():
            if id_ in self.ignore_labels:
                self.id2tag[id_] = 'O'
            else:
                self.id2tag[id_] = f'I-{nickname}'
        self.tokenizer = tokenizer
        self.save_annotation = save_annotation
        self.save_output = save_output
        self.return_annotation = return_annotation
        self.word_level = word_level
    
    def _merge_consecutive_two_token(self, first_input_id:int, second_input_id:int):
        first_input_id = int(first_input_id)
        second_input_id = int(second_input_id)
        if isinstance(self.tokenizer, transformers.RobertaTokenizerFast):
            return (not self.tokenizer.convert_ids_to_tokens(second_input_id).startswith(chr(288))) and \
                self.tokenizer.convert_ids_to_tokens(first_input_id)[-1].isalpha() and \
                self.tokenizer.convert_ids_to_tokens(second_input_id)[0].isalpha()
        elif isinstance(self.tokenizer, transformers.XLMRobertaTokenizerFast):
            return (not self.tokenizer.convert_ids_to_tokens(second_input_id).startswith(chr(9601))) and \
                self.tokenizer.convert_ids_to_tokens(first_input_id)[-1].isalpha() and \
                self.tokenizer.convert_ids_to_tokens(second_input_id)[0].isalpha()
        elif isinstance(self.tokenizer, transformers.BertTokenizerFast):
            return self.tokenizer.convert_ids_to_tokens(second_input_id).startswith('##')
    
    def is_middle_token(self, input_id:int):
        if isinstance(self.tokenizer, transformers.RobertaTokenizerFast): 
            return not self.tokenizer.convert_ids_to_tokens(input_id).startswith(chr(288)) and \
                self.tokenizer.convert_ids_to_tokens(input_id).isalpha()
        elif isinstance(self.tokenizer, transformers.XLMRobertaTokenizerFast):
            return not self.tokenizer.convert_ids_to_tokens(input_id).startswith(chr(9601)) and \
                self.tokenizer.convert_ids_to_tokens(input_id).isalpha()
        elif isinstance(self.tokenizer, transformers.BertTokenizerFast):
            return self.tokenizer.convert_ids_to_tokens(input_id).startswith('##')
    

    @classmethod
    def find_offsets(cls, seq_str:str, match:re.Pattern):
        annotations = []
        for annotation in match.finditer(seq_str):
            start = int(annotation.group('start'))
            label = annotation.group('label')
            end = annotation.group('end')
            end = start + 1 if end is None else int(end) + 1
            annotations.append((start, end, label))
        return annotations
    
    def collect_spans(self, sequence:str) -> Set[Tuple[int, int, str]]:
        spans = self.find_offsets(sequence, self.IO_match)
        label_spans = set()
        for span in spans:
            label_spans.add((span[0], span[1], self.nickname2label[span[2]]))
        return label_spans

    def _preprocess(self, array:Union[List[Union[List[List[int]], List[int], torch.Tensor, np.ndarray]], torch.Tensor, np.ndarray]):
        if isinstance(array, list):
            if isinstance(array[0], list):
                if isinstance(array[0][0], list):
                    array = [np.array(sequence) for batch in array for sequence in batch]
                else:
                    array = [np.array(sequence) for sequence in array]
            elif isinstance(array[0], np.ndarray):
                if len(array[0].shape) == 2:
                    array = [sequence for batch in array for sequence in batch]
                elif len(array[0].shape) == 1:
                    pass
                else:
                    raise ValueError(f"Cannot parse List of ndarray of shape {array[0].shape}.")
            elif isinstance(array[0], torch.Tensor):
                if len(array[0].size()) == 2:
                    array = [sequence.numpy() for batch in array for sequence in batch]
                elif len(array[0].shape) == 1:
                    array = [sequence.numpy() for sequence in array]
                else:
                    raise ValueError(f"Cannot parse List of pytorch tensor of size {array[0].size()}.")
        elif isinstance(array, np.ndarray):
            pass
        elif isinstance(array, torch.Tensor):
            array = array.numpy()
        sequences = []
        for idx, sequence in enumerate(array):
            sequence = sequence[sequence!=self.pad_value]
            sequences.append(" ".join([f'{offset}{self.id2tag[token]}' for offset, token in enumerate(sequence)]) + " ")
        return sequences
    
    def annotate(self, predictions:List[Union[List[Tuple[int, int, str]], Set[Tuple[int, int, str]]]], encodings:Union[List[BatchEncoding], BatchEncoding],save:bool=True) -> None:
        fw = None
        if save: fw = open(os.path.join(self.save_dir, self.save_name), "wt")
        corpus_annotations = []
        for i, prediction in enumerate(predictions):
            if isinstance(encodings, list):
                encoding = encodings[i]
            annotations = []
            for annotation in prediction:
                # print(annotation)
                start_pt = annotation[0]
                end_pt = annotation[1]
                try:
                    if self.word_level:
                        if isinstance(encodings, list):
                            start = encoding.token_to_word(start_pt)
                            if start is None:
                                continue
                            end = encoding.token_to_word(end_pt-1)
                            if end is None:
                                continue
                            else:
                                end += 1
                        else:
                            start = encodings.token_to_word(i, start_pt)
                            if start is None:
                                continue
                            end = encodings.token_to_word(i, end_pt-1)
                            if end is None:
                                continue
                            else:
                                end += 1
                    else:
                        if isinstance(encodings, list):
                            start = encoding.token_to_chars(start_pt)
                            if start is not None:
                                start = start.start
                            else:
                                continue
                            end = encoding.token_to_chars(end_pt-1)
                            if end is not None:
                                end = end.end
                            else:
                                continue
                        else:
                            start = encodings.token_to_chars(i, start_pt)
                            if start is not None:
                                start = start.start
                            else:
                                continue
                            end = encodings.token_to_chars(i, end_pt-1)
                            if end is not None:
                                end = end.end
                            else:
                                continue
                    annotations.append([start, end, annotation[2]])
                except TypeError as e:
                    print(e)
                    continue
            if save:
                fw.write(json.dumps({"annotations": annotations})+"\n")
            corpus_annotations.append(annotations)
        if save: fw.close()
        return corpus_annotations
    
    def fix_spans(self, predictions:List[Union[List[Tuple[int, int, str]], Set[Tuple[int, int, str]]]], encodings:List[BatchEncoding]) -> List[Set[Tuple[int, int, str]]]:
        fixed = []
        for i, prediction in enumerate(predictions):
            encoding = encodings[i]
            input_ids = getattr(encoding, 'input_ids', getattr(encoding, 'ids', None))
            annotations = set()
            for annotation in prediction:
                start_pt = annotation[0]
                end_pt = annotation[1]
                if start_pt >= len(input_ids) - 1 or start_pt == 0 or end_pt >= len(input_ids):
                    continue
                if self.fix_span == 'extend':
                    while start_pt > 1 and self._merge_consecutive_two_token(input_ids[start_pt-1], input_ids[start_pt]):
                        start_pt -= 1
                    while end_pt < len(input_ids) - 1 and self._merge_consecutive_two_token(input_ids[end_pt-1], input_ids[end_pt]):
                        end_pt += 1
                    annotations.add((start_pt, end_pt, annotation[2]))
                elif self.fix_span == 'head':
                    if all([self.is_middle_token(input_ids[pt]) for pt in range(start_pt, end_pt)]):
                        continue
                    while start_pt < len(input_ids) - 1 and self.is_middle_token(input_ids[start_pt]):
                        start_pt += 1
                    while end_pt < len(input_ids) and self.is_middle_token(input_ids[end_pt]):
                        end_pt += 1
                    annotations.add((start_pt, end_pt, annotation[2]))
                else:
                    raise ValueError("fix_span must be one of ['none', 'extend', 'head']")
                    
            fixed.append(annotations)
        return fixed
    
    def _get_fn_errors(self, prediction_spans:Set, label_spans: Set):
        spans = {t[:3]: t[3] for t in prediction_spans}
        missed = label_spans.difference(prediction_spans)
        fn = set()
        for m in missed:
            if m[:3] in spans:
                fn.add(m + (spans[m[:3]],))
            else:
                fn.add(m + ('NA',))
        return fn

    def __call__(self, outputs:Dict[str, Any], encodings:Optional[List[BatchEncoding]]=None) -> Dict[str, float]:
        predictions = outputs['prediction']
        predictions = self._preprocess(predictions)
        predictions = [self.collect_spans(prediction) for prediction in predictions]

        labels = None
        if 'label' in outputs:
            labels = outputs['label']
            if labels is not None:
                labels = self._preprocess(labels)
                labels = [self.collect_spans(label) for label in labels]
        
        if self.fix_span != 'none' and self.tokenizer is not None and encodings is not None:
            predictions = self.fix_spans(predictions, encodings)
        
        metric = F1Record()
        metric_by_label = {}
        macro = {}
        fn_errors = set()
        if self.candidates is not None and len(self.candidates) == len(predictions):
            assert len(self.candidates) == len(predictions)
            nmatch = 0; nlabel = 0; nprediction = 0
            for cands, preds in zip(self.candidates, predictions):
                for start, end, type_ in cands:
                    ctype = 'NA'
                    for pstart, pend, ptype in preds:
                        if start >= pstart and end <= pend:
                            ctype = ptype
                            break
                    if type_ != 'NA':
                        nlabel += 1
                        if type_ not in metric_by_label:
                            metric_by_label[type_] = [1, 0, 0]
                        else:
                            metric_by_label[type_][0] += 1
                    if ctype != 'NA':
                        nprediction += 1
                        if ctype not in metric_by_label:
                            metric_by_label[ctype] = [0, 1, 0]
                        else:
                            metric_by_label[ctype][1] += 1
                    if type_ == ctype and type_ != 'NA': 
                        nmatch += 1
                        metric_by_label[type_][2] += 1
            metric += np.array([nlabel, nprediction, nmatch]) 
            metric_by_label = {k: F1Record(v) for k, v in metric_by_label.items()}
            macro = {k: sum(m.named_result[k] for l,m in metric_by_label.items()) / len(metric_by_label) for k in ['precision', 'recall', 'f1']}
            macro = MacroF1Record(macro)
        elif labels is not None:
            prediction_spans = {tuple([i]+list(prediction_span)) for i, prediction_spans in enumerate(predictions) for prediction_span in prediction_spans}
            label_spans = {tuple([i]+list(label_span)) for i, label_spans in enumerate(labels) for label_span in label_spans}
            nprediction = len(prediction_spans)
            nlabel = len(label_spans)
            nmatch = len(prediction_spans.intersection(label_spans))
            
            metric += np.array([nlabel, nprediction, nmatch])      
            for label in self.label2id:
                ffunc = lambda t:t[3]==label
                nprediction = len(list(filter(ffunc, prediction_spans)))
                nlabel = len(list(filter(ffunc, label_spans)))
                nmatch = len(list(filter(ffunc, prediction_spans.intersection(label_spans))))
                if nlabel > 0:
                    metric_by_label[label] = F1Record()
                    metric_by_label[label] += np.array([nlabel, nprediction, nmatch])
            macro = {k: sum(m.named_result[k] for l,m in metric_by_label.items()) / len(metric_by_label) for k in ['precision', 'recall', 'f1']}
            macro = MacroF1Record(macro)
        annotations = None
        if self.save_dir is not None:
            if self.save_output:
                save_output = os.path.join(self.save_dir, "output.th")
                torch.save({'prediction': predictions, 'labels': labels, 'metric': metric, 'metric_by_label': metric_by_label, 'macro': macro}, save_output)
                print(f"save to {save_output}")
            if self.save_annotation and encodings is not None:
                annotations = self.annotate(predictions, encodings, save=True)
        if annotations is None and self.return_annotation:
            annotations = self.annotate(predictions, encodings, save=False)

        if self.return_annotation:
            return MixF1Record(metric, macro, 'MIC', 'MAC'), metric_by_label, MixF1Record(macro, metric, 'MAC', 'MIC'), annotations
        else:
            return MixF1Record(metric, macro, 'MIC', 'MAC'), metric_by_label, MixF1Record(macro, metric, 'MAC', 'MIC')


def simpleF1(outputs, *args, **kwargs):
        pred = outputs["prediction"]
        label = outputs["label"]
        pred = torch.cat(pred, dim=0)
        label = torch.cat(label, dim=0)
        valid = (label > 0).long()
        nmatch = torch.sum((pred == label).long() * valid)
        ngold = torch.sum(valid)
        npred = torch.sum((pred > 0).long())
        record = F1Record()
        record += np.array((ngold.item(), npred.item(), nmatch.item()))
        return record, None

class WSAnnotations(object):
    
    def __init__(self, threshold:float, tokenizer:PreTrainedTokenizerFast, label2id:Dict[str, int], uthreshold:Optional[float]=None, id2label:Optional[Dict[int, str]]=None, save_dir:Optional[str]=None):
        self.threshold = threshold
        if uthreshold is None:
            self.uthreshold = threshold
        else:
            self.uthreshold = uthreshold
        self.tokenizer = tokenizer
        self.label2id = label2id
        if id2label is None:
            self.id2label = {v:k for k,v in label2id.items()}
            self.label2id['CND'] = len(label2id)
            self.id2label[len(label2id)] = 'CND'
        else:
            self.id2label = id2label
            self.label2id['CND'] = max(len(label2id), len(id2label))
            self.id2label[max(len(label2id), len(id2label))] = 'CND'
        self.rlabel2id = {v:k for k,v in label2id.items()}
        self.f1_tagger = F1MetricTag(
            pad_value=-100,
            ignore_labels=0,
            label2id=self.label2id,
            tokenizer=self.tokenizer,
            save_dir=save_dir,
            save_output=False,
            save_annotation=False,
            return_annotation=True)
        
    
    def collect(self, pieces, pred):
        spans = []
        span = []
        for it in range(pred.size(0)):
            if pred[it] == -100:
                if len(span) == 3:
                    spans.append(tuple(span))
                break
            if pred[it] > 0:
                if len(span) == 0:
                    if pieces[it].startswith('##'):
                        continue
                    span = [int(pred[it]), it, it+1]
                elif span[0] == pred[it]:
                    span[2] = it+1
                else:
                    if pieces[it].startswith('##'):
                        span[2] = it + 1
                    else:
                        spans.append(tuple(span))
                        span = [int(pred[it]), it, it+1]
            else:
                if len(span) == 3:
                    if pieces[it].startswith('##'):
                        span[2] = it + 1
                    else:
                        spans.append(tuple(span))
                        span = []
        return spans
    
    def annotate(self, input_ids:Union[torch.LongTensor, List[transformers.BatchEncoding]], outputs:Optional[torch.FloatTensor]=None, tokenizer:Optional[PreTrainedTokenizerFast]=None, return_offset:Optional[str]='char'):
        encodings = None
        if outputs is None:
            input_ids, outputs = input_ids
        if tokenizer is None:
            tokenizer = self.tokenizer
        if isinstance(input_ids, transformers.BatchEncoding):
            encodings = input_ids
            input_ids = encodings.input_ids
        elif isinstance(input_ids[0], transformers.BatchEncoding):
            encodings = input_ids
            input_ids = self.tokenizer.pad(encodings, return_tensors='pt').input_ids
        if outputs.size(1) > input_ids.size(1):
            outputs = outputs[:, :input_ids.size(1), :]
        
        val, pred = torch.max(outputs, dim=-1)
        mask = val > self.threshold
        umask = (val <= self.threshold) & (val > self.uthreshold)
        pred = pred + 1
        pred[~mask] = 0
        pred[umask] = self.label2id['CND']

        for idx, label in self.id2label.items():
            pred[pred==idx] = self.label2id[label]

        if tokenizer.pad_token_id is not None:
            pred[input_ids==tokenizer.pad_token_id] = -100
        if tokenizer.cls_token_id is not None:
            pred[input_ids==tokenizer.cls_token_id] = 0
        if tokenizer.sep_token_id is not None:
            pred[input_ids==tokenizer.sep_token_id] = -100
        
        if encodings is not None:
            metric, _, annotations = self.f1_tagger({'prediction': pred}, encodings=encodings)
            instances = [{'tokens':None, 'annotations': annotation} for annotation in annotations]
        else:
            instances = []
            for i in range(input_ids.size(0)):
                pieces = tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
                output = pred[i, 1:len(pieces)+1]
                span = self.collect(pieces, output)
                sentence = ''
                index = 0
                annotations = []
                for label_id, start, end in span:
                    if index >= 0 and index < start:
                        if index > 0:
                            sentence += " "
                        sentence += tokenizer.convert_tokens_to_string(pieces[index:start])
                    text = tokenizer.convert_tokens_to_string(pieces[start:end])
                    annotations.append([len(sentence)+1, len(sentence) + len(text) + 1, self.rlabel2id[label_id], text])
                    sentence += f" {text}"
                    index = end
                if index < len(pieces):
                    if index > 0:
                        sentence += " "
                    sentence += tokenizer.convert_tokens_to_string(pieces[index:])
                instances.append({'tokens': sentence, 'annotations': annotations})
        return instances