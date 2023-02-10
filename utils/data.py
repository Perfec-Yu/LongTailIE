import argparse
from dataclasses import dataclass
from typing import *
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
import json
import os
from transformers import PreTrainedTokenizerFast, AutoTokenizer, BatchEncoding
from transformers.tokenization_utils_base import TokenSpan

_MUTUAL_RELATIONS = {"PHYS:Near", "PER-SOC:Business", "PER-SOC:Lasting-Personal", "PER-SOC:Family"}



@dataclass
class Annotation(object):
    triggers: List[Tuple[int, int, str]]
    entities: List[Tuple[int, int, str]]
    relations: List[Tuple[int, int, str]]
    roles: List[Tuple[int, int, str]]

    @classmethod
    def from_oneie(cls, oneie):
        entities = []
        entity_id_to_index = {}
        if 'entity_mentions' in oneie:
            for entity in oneie['entity_mentions']:
                if 'start' not in entity:
                    start = 0
                    end = 1
                else:
                    start = entity['start']
                    end = entity['end']
                if "entity_subtype" in entity:
                    label = entity['entity_subtype']
                else:
                    label = entity['entity_type']
                entity_id_to_index[entity['id']] = len(entities)
                entities.append((start, end, label))
        relations = []
        if 'relation_mentions' in oneie:
            for relation in oneie['relation_mentions']:
                relation_label = relation['relation_subtype'] if 'relation_subtype' in relation else relation["relation_type"]
                relations.append((
                    entity_id_to_index[relation['arguments'][0]['entity_id']],
                    entity_id_to_index[relation['arguments'][1]['entity_id']],
                    relation_label
                    ))
                if relation_label in _MUTUAL_RELATIONS:
                    relations.append((
                    entity_id_to_index[relation['arguments'][1]['entity_id']],
                    entity_id_to_index[relation['arguments'][0]['entity_id']],
                    relation_label
                    ))

        triggers = []
        roles = []
        if 'event_mentions' in oneie:
            for event in oneie['event_mentions']:
                start = event['trigger']['start']
                end = event['trigger']['end']
                label = event['event_type']
                triggers.append((start, end, label))
                if 'argument' in event:
                    for role in event['arguments']:
                        roles.append((
                            len(triggers) - 1,
                            entity_id_to_index[role['entity_id']],
                            role['role']
                        ))
        return cls(triggers=triggers, entities=entities, relations=relations, roles=roles)


@dataclass
class Instance(object):
    tokens : Union[List[str], str]
    annotations : Annotation
    sentence_id : str

    @classmethod
    def from_oneie(cls, oneie):
        if 'sentence' in oneie:
            tokens = oneie['sentence']
        else:
            tokens = oneie['tokens']
        annotations = Annotation.from_oneie(oneie)
        if 'sent_id' in oneie:
            sentence_id = oneie["sent_id"]
        else:
            sentence_id = '0'
        return cls(tokens=tokens, annotations=annotations, sentence_id=sentence_id)

def _to_instance(data, sentence_id_prefix:Optional[str]=None):
    if sentence_id_prefix is None:
        sentence_id_prefix = ""
    elif not sentence_id_prefix.endswith("_"):
        sentence_id_prefix += "_"
    if 'annotations' in data[0]:
        if 'sentence_id' in data[0]:
            for t in data:
                t["sentence_id"] = f"{sentence_id_prefix}{t['sentence_id']}"
            return [Instance(**t) for t in data]
        else:
            return [Instance(**t, sentence_id=f"{sentence_id_prefix}{i}") for i, t in enumerate(data)]
    else:
        instances = [Instance.from_oneie(t) for t in data]
        return instances

class IDataset(Dataset):
    _LABEL_TYPES = ['event', 'entity', 'role', 'relation']
    _SEED = 2147483647
    def __init__(self,
        instances:List[Instance],
        label2id:Dict[str, Dict[str, int]],
        tokenizer:PreTrainedTokenizerFast,
        max_length:Optional[int]=None,
        use_pseudo_labels:bool=False,
        seed:Optional[int]=None,
        label_ignore:Optional[Union[Dict, List, Set, int]]=None,
        root:Optional[str]=None,
        *args,
        **kwargs) -> None:
        super().__init__()
        if isinstance(label_ignore, int):
            label_ignore = {
                label_type: {label_ignore} 
                for label_type in self._LABEL_TYPES
            }
        elif isinstance(label_ignore, list) or isinstance(label_ignore, set):
            label_ignore = label_ignore = {
                label_type: set(label_ignore)
                for label_type in self._LABEL_TYPES
            }
        self.label_ignore = label_ignore if label_ignore else {
                label_type: set() 
                for label_type in self._LABEL_TYPES
            }
        self.label2id = label2id
        for k in self._LABEL_TYPES:
            if k not in self.label2id: self.label2id[k] = {"NA": 0}
        self.instances = instances
        self.instance_tokenized = isinstance(instances[0].tokens, list)
        self.tokenizer = tokenizer
        self.use_pseudo_labels = use_pseudo_labels
        self.max_length = max_length
        self.root = root
        self.seed = seed if seed else self._SEED
        self._generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        if self.root:
            ctx_feat = torch.load(os.path.join(self.root, "contextual", self.instances[index].sentence_id))
            return self.instances[index], ctx_feat
        else:
            return self.instances[index]
    
    def collate_batch(self, batch:List[Union[Instance, Tuple[Instance, torch.Tensor]]]) -> BatchEncoding:
        if self.root:
            ctx_feats = [t[1] for t in batch]
            batch = [t[0] for t in batch]
        text = []
        labels = None
        sentence_labels = None
        annotations = []
        spans = None
        for i in batch:
            text.append(i.tokens)
            annotations.append(i.annotations)
        encoded:BatchEncoding = self.tokenizer(
            text=text,
            max_length=self.max_length,
            is_split_into_words=self.instance_tokenized,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )
        _ = encoded.pop("special_tokens_mask")
        _n_triggers = max(len(t.triggers) if t is not None else 0 for t in annotations)
        trigger_spans = torch.zeros(len(batch), _n_triggers, encoded.input_ids.size(1), dtype=torch.float)
        trigger_labels = torch.zeros_like(encoded.input_ids, dtype=torch.long)
        trigger_labels[encoded['attention_mask']==0] = -100
        sentence_labels = torch.zeros(encoded.input_ids.size(0), max(self.label2id['event'].values()))
        _n_entities = max(len(t.entities) if t is not None else 0 for t in annotations)
        entity_spans = torch.zeros(len(batch), _n_entities, encoded.input_ids.size(1), dtype=torch.float)
        entity_labels = torch.zeros_like(encoded.input_ids, dtype=torch.long)
        _n_relations = sum([len(t.relations) if t is not None else 0 for t in annotations])
        relation_labels = - torch.ones(len(batch), _n_entities, _n_entities, dtype=torch.long)
        _n_roles = sum([len(t.roles) if t is not None else 0 for t in annotations if t is not None])
        role_labels = - torch.ones(len(batch), _n_triggers, _n_entities, dtype=torch.long)
        role_event_labels = - torch.ones(len(batch), _n_triggers, _n_entities, dtype=torch.long)

        if self.root:
            ctx_features = torch.zeros(encoded.input_ids.size(0), encoded.input_ids.size(1), ctx_feats[0].size(1))
            for i, ctx_feat in enumerate(ctx_feats):
                valid_len = min(ctx_feat.size(0), encoded.input_ids.size(1)-2)
                ctx_features[i, 1:valid_len+1] = ctx_feat[:valid_len]

        irel = irol = 0

        for ibatch, anns in enumerate(annotations):
            if anns is None:
                trigger_labels[ibatch, :] = -100
                sentence_labels[ibatch] = -100
                entity_labels[ibatch, :] = -100
                continue
            role_labels[ibatch, :len(anns.triggers), :len(anns.entities)] = 0
            role_event_labels[ibatch, :len(anns.triggers), :len(anns.entities)] = 0
            relation_labels[ibatch, :len(anns.entities), :len(anns.entities)] = 0
            relation_labels[ibatch, torch.arange(len(anns.entities)), torch.arange(len(anns.entities))] = -1
            for iann, ann in enumerate(anns.triggers):
                start, end, label = ann[:3]
                label_id = self.label2id['event'][label] if label in self.label2id['event'] else 0
                label_id = label_id if label_id not in self.label_ignore['event'] else 0
                if label_id > 0:
                    sentence_labels[ibatch, label_id-1] = 1
                if self.instance_tokenized:
                    tok_start = encoded.word_to_tokens(ibatch, start)
                    tok_end = encoded.word_to_tokens(ibatch, end)
                else:
                    if start < 0:
                        print(ann)
                    tok_start = encoded.char_to_token(ibatch, start)
                    tok_end = encoded.char_to_token(ibatch, end-1)
                if tok_end is not None and tok_start is not None:
                    if isinstance(tok_start, TokenSpan):
                        tok_start = tok_start.start
                    if isinstance(tok_end, TokenSpan):
                        tok_end = tok_end.start
                    else:
                        tok_end += 1
                    trigger_spans[ibatch, iann, tok_start:tok_end] = 1. / (tok_end - tok_start)
                    trigger_labels[ibatch, tok_start:tok_end] = label_id
            for iann, ann in enumerate(anns.entities):
                start, end, label = ann[:3]
                label_id = self.label2id['entity'][label] if label in self.label2id['entity'] else 0
                label_id = label_id if label not in self.label_ignore['entity'] else 0
                if self.instance_tokenized:
                    tok_start = encoded.word_to_tokens(ibatch, start)
                    tok_end = encoded.word_to_tokens(ibatch, end)
                else:
                    tok_start = encoded.char_to_token(ibatch, start)
                    tok_end = encoded.char_to_token(ibatch, end-1)
                if tok_end is not None and tok_start is not None:
                    if isinstance(tok_start, TokenSpan):
                        tok_start = tok_start.start
                    if isinstance(tok_end, TokenSpan):
                        tok_end = tok_end.start
                    else:
                        tok_end += 1
                    if tok_end == tok_start:
                        print(start, end, text[ibatch])
                    entity_spans[ibatch, iann, tok_start:tok_end] = 1. / (tok_end - tok_start)
                    entity_labels[ibatch, tok_start:tok_end] = label_id
            for iann, ann in enumerate(anns.relations):
                label_id = self.label2id['relation'][ann[2]] if ann[2] in self.label2id['relation'] else 0
                relation_labels[ibatch, ann[0], ann[1]] = label_id if label_id not in self.label_ignore['relation'] else 0
                irel += 1
            for iann, ann in enumerate(anns.roles):
                label_id = self.label2id['role'][ann[2]] if ann[2] in self.label2id['role'] else 0
                role_labels[ibatch, ann[0], ann[1]] = label_id if label_id not in self.label_ignore['role'] else 0
                event_label_id = self.label2id['event'][anns.triggers[ann[0]][2]]
                role_event_labels[ibatch, ann[0], ann[1]] = event_label_id if event_label_id not in self.label_ignore['event'] else 0
                irol += 1
        
        prefix = "pseudo_" if self.use_pseudo_labels else ""
        encoded[prefix+"trigger_labels"] = trigger_labels
        # encoded[prefix+"trigger_spans"] = trigger_spans
        # encoded[prefix+"sentence_labels"] = sentence_labels
        encoded[prefix+"entity_labels"] = entity_labels
        # encoded[prefix+"entity_spans"] = entity_spans
        # encoded[prefix+"relation_labels"] = relation_labels
        # encoded[prefix+"role_labels"] = role_labels
        # encoded[prefix+"role_event_labels"] = role_event_labels
        if self.root:
            encoded[prefix+"context_features"] = ctx_features
        return encoded

    def collate_fn(self, batch:List[Union[Instance, Tuple[Instance, torch.Tensor]]]) -> BatchEncoding:
        input_batch = self.collate_batch(batch)

        remove_keys = [key for key in input_batch if input_batch[key] is None]
        for key in remove_keys:
            input_batch.pop(key)
        return input_batch

'''
def get_dev_test_encodings(
    opts:Optional[argparse.Namespace]=None,
    root:Optional[str]=None,
    dataset:Optional[str]=None,
    model_name:Optional[str]=None,
    setting:Optional[str]=None,
    max_length:Optional[int]=None,
    test_only:Optional[bool]=None,
    seed:Optional[int]=None,
    *args,
    **kwargs):

    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        seed = getattr(opts, "seed", None) if seed is None else seed
    dev_file = os.path.join(root, DEV_FILE)
    test_file = os.path.join(root, TEST_FILE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(dev_file, "rt") as f:
        dev = _to_instance([json.loads(line) for line in f], "dev", True, False)
    with open(test_file, "rt") as f:
        test = _to_instance([json.loads(line) for line in f], "test", True, False)
    dev = [tokenizer(
            text=t.tokens,
            max_length=max_length,
            is_split_into_words=isinstance(t.tokens, list),
            add_special_tokens=True,
            padding='longest',
            truncation=True) for t in dev]
    test = [tokenizer(
            text=t.tokens,
            max_length=max_length,
            is_split_into_words=isinstance(t.tokens, list),
            add_special_tokens=True,
            padding='longest',
            truncation=True) for t in test]
    return dev, test

def get_datasets(
    opts:Optional[argparse.Namespace]=None,
    root:Optional[str]=None,
    dataset:Optional[str]=None,
    model_name:Optional[str]=None,
    setting:Optional[str]=None,
    max_length:Optional[int]=None,
    parallel:Optional[bool]=None,
    test_only:Optional[bool]=None,
    seed:Optional[int]=None,
    *args,
    **kwargs) -> Tuple[Union[IDataset, None], Union[IDataset, None], IDataset]:

    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        parallel = getattr(opts, "parallel", None) if parallel is None else parallel
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        seed = getattr(opts, "seed", None) if seed is None else seed
    label_info_file = os.path.join(root, "label_info.json")
    train_file = os.path.join(root, TRAIN_FILE)
    trans_train_file = os.path.join(root, TRANS_TRAIN_FILE)
    train_no_ann_file = os.path.join(root, TRAIN_NO_ANN_FILE)
    trans_train_no_ann_file = os.path.join(root, TRANS_TRAIN_NO_ANN_FILE)
    dev_file = os.path.join(root, DEV_FILE)
    trans_dev_file = os.path.join(root, TRANS_DEV_FILE)
    test_file = os.path.join(root, TEST_FILE)
    trans_test_file = os.path.join(root, TRANS_TEST_FILE)
    print("train:", train_file)
    print("dev:", dev_file)
    print("test:", test_file)
    load_trans = parallel
    load_train_dev_file = not test_only
    build_train_dev_dataset = not test_only

    label_info = label2id = None
    train = dev = test = None

    train_dataset = None
    train_no_ann_dataset = None
    dev_dataset = None

    print("loading files...")
    with open(label_info_file, "rt") as f:
        label_info = json.load(f)
        label2id = {
            label_type: {
                label: info["id"] 
                for label,info in type_info.items()
            }
            for label_type, type_info in label_info.items()
        }
        for label_type in label2id:
            label2id[label_type]["NA"] = 0
        
    if load_train_dev_file:
        with open(train_file, "rt") as f:
            train = _to_instance([json.loads(line) for line in f], "train", True, False)
        with open(train_no_ann_file, "rt") as f:
            train_no_ann = _to_instance([json.loads(line) for line in f], "train", True, False)
        with open(dev_file, "rt") as f:
            dev = _to_instance([json.loads(line) for line in f], "dev", True, False)
        if load_trans:
            with open(trans_train_file, "rt") as f:
                train_t = _to_instance([json.loads(line) for line in f], "train", True, False)
            with open(trans_train_no_ann_file, "rt") as f:
                train_no_ann_t = _to_instance([json.loads(line) for line in f], "train", False, False)
            with open(trans_dev_file, "rt") as f:
                dev_t = _to_instance([json.loads(line) for line in f], "dev", False, False)
            with open(trans_test_file, "rt") as f:
                test_t = _to_instance([json.loads(line) for line in f], "test", False, False)
    elif load_trans:
        with open(trans_test_file, "rt") as f:
            test_t = _to_instance([json.loads(line) for line in f], "test", False, False)

    
    with open(test_file, "rt") as f:
        test = _to_instance([json.loads(line) for line in f], "test", True, False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("building pytorch datasets...")
    if build_train_dev_dataset:
        if parallel:
            train_dataset = ParallelDataset(
                instances=[train, train_t] if TRAIN_FILE.startswith("en") else [train_t, train],
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=max_length,
                mask_prob=None,
                eval_lang=None,#0 if TRAIN_FILE.startswith("en") else 1,
                seed=seed)
            if ADD_NO_ANN:
                train_no_ann_dataset = ParallelDataset(
                    instances=[train_no_ann, train_no_ann_t] if TRAIN_NO_ANN_FILE.startswith("en") else [train_no_ann_t, train_no_ann],
                    label2id=label2id,
                    tokenizer=tokenizer,
                    use_pseudo_labels=False,
                    max_length=max_length,
                    mask_prob=None,
                    seed=seed)
            dev_dataset = ParallelDataset(
                instances=[dev_t, dev],
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=max_length,
                eval_lang=1)
        else:
            train_dataset = IDataset(
                instances=train,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=max_length,
                mask_prob=None,
                seed=seed)
            dev_dataset = IDataset(
                instances=dev,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=max_length)
    test_dataset = (ParallelDataset if parallel else IDataset)(
        instances=[test_t, test] if parallel else test,
        label2id=label2id,
        tokenizer=tokenizer,
        max_length=max_length,
        eval_lang=1)
    
    if train_no_ann_dataset:
        return [train_dataset, train_no_ann_dataset], dev_dataset, test_dataset
    else:
        return train_dataset, dev_dataset, test_dataset

def get_data(
    opts:Optional[argparse.Namespace]=None,
    batch_size:Optional[int]=None,
    eval_batch_size:Optional[int]=None,
    num_workers:Optional[int]=None,
    seed:Optional[int]=None,
    root:Optional[str]=None,
    dataset:Optional[str]=None,
    model_name:Optional[str]=None,
    setting:Optional[str]=None,
    max_length:Optional[int]=None,
    parallel:Optional[bool]=None,
    test_only:Optional[bool]=None,
    shuffle:Optional[bool]=None,
    *args,
    **kwargs):
    _default_num_workers = 0
    _default_seed = 44739242
    if opts is not None:
        root = getattr(opts, "root", None) if root is None else root
        dataset = getattr(opts, "dataset", None) if dataset is None else dataset
        batch_size = getattr(opts, "batch_size", None) if batch_size is None else batch_size
        eval_batch_size = getattr(opts, "eval_batch_size", batch_size) if eval_batch_size is None else eval_batch_size
        num_workers = getattr(opts, "num_workers", _default_num_workers) if num_workers is None else num_workers
        seed = getattr(opts, "seed", _default_seed) if seed is None else seed
        model_name = getattr(opts, "model_name", None) if model_name is None else model_name
        setting = getattr(opts, "setting", None) if setting is None else setting
        max_length = getattr(opts, "max_length", None) if max_length is None else max_length
        test_only = getattr(opts, "test_only", None) if test_only is None else test_only
        parallel = getattr(opts, "parallel", None) if parallel is None else parallel
    if shuffle is None:
        shuffle = True
    with open(os.path.join(root, "label_info.json"), "rt") as f:
        label_info = json.load(f)
        label2id = {
            label_type: {
                label: info["id"] 
                for label,info in type_info.items()
            }
            for label_type, type_info in label_info.items()
        }
        for label_type in label2id:
            label2id[label_type]["NA"] = 0

    datasets = get_datasets(
        root=root,
        dataset=dataset,
        model_name=model_name,
        setting=setting,
        max_length=max_length,
        seed=seed,
        parallel=parallel,
        test_only=test_only)

    if test_only:
        loaders = [None] * (len(datasets) - 1)
    else:
        loaders = []
        if isinstance(datasets[0], list):
            loaders.append([DataLoader(
                dataset=d,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                collate_fn=d.collate_fn,
                pin_memory=True,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(seed)) for d in datasets[0]])
        else:
            loaders.append(DataLoader(
                dataset=datasets[0],
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                collate_fn=datasets[0].collate_fn,
                pin_memory=True,
                num_workers=num_workers,
                generator=torch.Generator().manual_seed(seed)
            ))
        loaders.extend([DataLoader(
            dataset=d,
            batch_size=batch_size if eval_batch_size <= 0 else eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=d.collate_fn,
            pin_memory=True,
            num_workers=num_workers) for d in datasets[1:-1]])
    test_loader = DataLoader(
        dataset=datasets[-1],
        batch_size=batch_size if eval_batch_size <= 0 else eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=datasets[-1].collate_fn,
        pin_memory=True,
        num_workers=num_workers)
    loaders.append(test_loader)
    return loaders, label2id
'''