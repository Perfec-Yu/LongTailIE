from collections import defaultdict
import argparse
import warnings
import torch
import os
import logging
import time
import datetime
import numpy as np
import torch
from tqdm import tqdm
from typing import *

from transformers import AutoTokenizer

from .utils import F1Record, Record, create_optimizer_and_scheduler, F1MetricTag
from .data import _to_instance, IDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data import DataLoader

import json




class Worker(object):

    def __init__(self, opts, make_log_dir=True):
        super().__init__()
        self.no_gpu = opts.no_gpu
        self.gpu = opts.gpu
        self.gpu_ids = [int(t.strip()) for t in opts.gpu.strip().split(",")]
        self.distributed = len(self.gpu_ids) > 1
        self.log_dir = opts.log_dir
        if not os.path.exists(self.log_dir) and make_log_dir:
            os.makedirs(self.log_dir)
            
        else:
            self.log = None
        if os.path.exists(self.log_dir):
            now = str(datetime.datetime.now()).replace('-', '').replace(':', '').replace(' ', '.')
            self.log = os.path.join(self.log_dir, f"{now}.log")
            logging.basicConfig(filename=self.log, level=logging.INFO)
        self._log = logging.info
        self.save_model = os.path.join(self.log_dir, f"model")
        self.load_model = os.path.join(self.log_dir, f"model")

        self.train_epoch = opts.train_epoch
        self.train_step = opts.train_step
        self.accumulation_steps = opts.accumulation_steps
        self.weight_decay = opts.weight_decay
        self.learning_rate = opts.learning_rate
        self.max_grad_norm = opts.max_grad_norm
        self.warmup = opts.warmup

        self.epoch = 0
        self.accumulated_steps = 0
        self.optimization_step = 0
        self.train_iterator = None
        self.train_last_it = -1
        self.patience = opts.patience
        self.min_epoch = opts.min_epoch

        self.root = opts.root
        self.run_method = opts.run_method
        self.model_name = opts.model_name
        self.batch_size = opts.batch_size
        self.eval_batch_size = opts.eval_batch_size
        self.max_length = opts.max_length
        self.num_workers = opts.num_workers
        self.word_level = opts.word_level
        self.add_test = opts.add_test

        self.net_module = None
        self.net_args = None
        self.output_fn = None
        self.output_args = None

    @classmethod
    def from_options(cls, no_gpu:bool, gpu:int, log_dir:str, train_epoch:int=-1, train_step:int=-1, accumulation_steps:int=1, weight_decay:float=0., learning_rate:float=1e-5, max_grad_norm:float=1., warmup:float=0):
        class Opts:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        opts = Opts(
            no_gpu=no_gpu,
            gpu=gpu,
            log_dir=log_dir,
            train_epoch=train_epoch,
            train_step=train_step,
            accumulation_steps=accumulation_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            warmup=warmup)
        return cls(opts)

    @classmethod
    def _to_device(cls, instance:Union[torch.Tensor,List[torch.Tensor],Tuple[torch.Tensor,...],Dict[Any,torch.Tensor]], device:Union[torch.device, None]=None):
        if isinstance(instance, list):
            return [cls._to_device(t, device) for t in instance]
        elif isinstance(instance, dict):
            return {key: cls._to_device(value, device=device) for key, value in instance.items()}
        elif isinstance(instance, tuple):
            vals = [cls._to_device(value, device=device) for value in instance]
            return type(instance)(*vals)
        elif isinstance(instance, int) or isinstance(instance, float) or isinstance(instance, str):
            return instance
        else:
            try:
                return instance.to(device)
            except Exception as e:
                print(f"{type(instance)} not recognized for cuda")
                raise(e)


    def initialize(self, rank:int=-1, test_only:bool=False, seed:int=42):

        torch.manual_seed(seed)
        model = self.net_module(*self.net_args["args"], **self.net_args["kwargs"]).to(torch.device(f"cuda:{self.gpu_ids[rank]}"))
        if self.distributed:
            assert rank >= 0
            model = DDP(model, device_ids=[self.gpu_ids[rank]], output_device=self.gpu_ids[rank], find_unused_parameters=True)
        datasets = self.get_datasets(test_only)
        loaders = self.get_dataloaders(
            train_dataset=datasets["train"],
            dev_dataset=datasets["dev"],
            test_dataset=datasets["test"],
            collate_fn=None,
            eval_collate_fn=None,
            seed=seed
            )
        output_fn = None if self.output_fn is None else lambda model, batch: self.output_fn(model, batch, *self.output_args["args"], **self.output_args["kwargs"])
        misc = self.get_misc()
        if self.train_step < 0:
            self.train_step = misc["epoch_steps"] * self.train_epoch
        warmup_step = int(self.train_step * self.warmup)
        optimizer, scheduler = create_optimizer_and_scheduler(model, self.learning_rate, self.weight_decay, warmup_step, self.train_step)

        return model, output_fn, optimizer, scheduler, loaders, misc

    def set_model_params(self, net_module:torch.nn.Module, *args, **kwargs):
        self.net_module = net_module
        self.net_args = {"args": args, "kwargs": kwargs}

    def set_output_params(self, output_fn:Callable, *args, **kwargs):
        self.output_fn = output_fn
        self.output_args = {"args": args, "kwargs": kwargs}

    def _train_step(self, model, batch, optimizer, scheduler=None, output_fn=None):
        output = output_fn(model, batch) if output_fn else model(batch)
        loss = output.pop("loss")
        if len(loss.size()) >= 1:
            loss = loss.mean()
        loss.backward()
        self.accumulated_steps += 1
        if self.accumulated_steps == self.accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            self.accumulated_steps = 0
            self.optimization_step += 1
            if scheduler:
                scheduler.step()
        return loss, output

    def _get_next(self, iterator, storage):
        item = None
        try:
            item = next(iterator)
        except StopIteration as e:
            iterator = iter(storage)
        return item, iterator

    def train(self, model, loader, optimizer, scheduler=None, output_fn=None, eval_loader=None, eval_metric=None, eval_args=None, progress=None, rank:int=-1, save_postfix:str=""):
        model.train()
        if not self.distributed:
            rank = -1
        it = iter(loader)
        epoch_loss = Record()
        self.accumulated_steps = 0
        self.optimization_step = 0
        self.epoch = 0
        eval_result = [0.1, 0]
        close_progress = False
        if not self.distributed and progress is None:
            close_progress = True
            progress = tqdm(desc="Epoch 0", total=self.train_step, ncols=100)
        no_better = 0
        while self.optimization_step < self.train_step + 1:
            batch, it = self._get_next(it, loader)
            if batch is None:
                self.epoch += 1
                if self.distributed:
                    with progress.get_lock():
                        desc = progress.desc
                        current_epoch = int(desc.strip().strip(":").split()[1])
                        progress.set_description(f"Epoch {current_epoch+1}")
                else:
                    progress.set_description(f"Epoch {self.epoch}")

                if self.distributed and rank > 0:
                    continue
                if eval_loader is not None and eval_metric is not None:
                    metric = self.eval(model, eval_loader, eval_metric, eval_args)
                    if metric[0] > eval_result[0]:
                        eval_result[0] = metric[0]
                        if len(metric) > 1:
                            eval_result[1] = metric[1]
                        if not self.distributed or rank == 0:
                            self.save(model, optimizer, scheduler, postfix=save_postfix)
                        print("Current Best", "|".join([str(t) for t in eval_result]))
                        no_better = 0
                    else:
                        if self.epoch > self.min_epoch:
                            no_better += 1
                        if no_better == self.patience:
                            break
                    model.train()
                continue
            if self.distributed:
                batch = self._to_device(batch, torch.device(f"cuda:{self.gpu_ids[rank]}"))
            else:
                batch = self._to_device(batch, torch.device(f"cuda:{self.gpu_ids[0]}"))
            loss, _ = self._train_step(model, batch, optimizer, scheduler, output_fn)
            if loss is not None:
                epoch_loss += loss.item()
            if not self.distributed or rank == 0:
                if self.accumulated_steps == 0:
                    progress.update(1)
                postfix = {"loss": f"{epoch_loss}"}
                progress.set_postfix(postfix)
        if close_progress:
            progress.close()

    def eval(self, model, loader, eval_metric, eval_args):
        model.eval()
        if not isinstance(loader, list):
            loader = [loader]
        results = []
        with torch.no_grad():
            for idx, split_loader in enumerate(loader):
                epoch_outputs = defaultdict(list)
                epoch_loss = Record()
                progress = tqdm(split_loader, desc=f"Eval {idx}", ncols=100)
                for batch in progress:
                    outputs = model(batch=self._to_device(batch, torch.device(f"cuda:{self.gpu_ids[0]}")), predict=True)
                    loss = outputs.pop("loss")
                    if isinstance(loss, float):
                        epoch_loss += loss
                    else:
                        epoch_loss += loss.item()
                    postfix = {"loss": f"{epoch_loss}"}
                    progress.set_postfix(postfix)
                    for key, val in outputs.items():
                        epoch_outputs[key].append(val)
                metrics = eval_metric(epoch_outputs, eval_args[idx])
                for output_log in [print, self._log]:
                    output_log(f"Eval_{idx}: {metrics}")
                results.append(metrics)
        return results

    def get_history(self, model, loader, feat_func):
        model.eval()
        with torch.no_grad():
            progress = tqdm(loader, desc=f"Computing Features", ncols=100)
            for batch in progress:
                model.update_history(batch=self._to_device(batch, torch.device(f"cuda:{self.gpu_ids[0]}")))
            output = feat_func(model)
        return output

    def save(self,
        model:Union[torch.nn.Module, Dict],
        optimizer:Union[torch.optim.Optimizer, Dict, None]=None,
        scheduler:Union[torch.optim.lr_scheduler._LRScheduler, Dict, None]=None,
        postfix:str=""):

        save_dirs = self.log_dir
        if not os.path.exists(save_dirs):
            os.makedirs(save_dirs)
        def get_state_dict(x):
            if x is None:
                return None
            elif isinstance(x, dict):
                return x
            else:
                try:
                    return x.state_dict()
                except Exception as e:
                    raise ValueError(f"model, optimizer or scheduler to save must be either a dict or have callable state_dict method")
        if postfix != "":
            save_model = os.path.join(save_dirs, f"{self.save_model}.{postfix}")
        else:
            save_model = os.path.join(save_dirs, self.save_model)
        torch.save({
            "state_dict": get_state_dict(model),
            "optimizer_state_dict": get_state_dict(optimizer),
            "scheduler_state_dict": get_state_dict(scheduler),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            "iter": self.epoch + 1
            },
            save_model
        )

    def load(self, model:torch.nn.Module, optimizer:Union[torch.optim.Optimizer, None]=None, scheduler:Union[torch.optim.lr_scheduler._LRScheduler,None]=None, path:Union[str, None]=None, load_iter:bool=True, strict:bool=True, load_params:Optional[set]=None) -> None:
        if path is None:
            path = self.load_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"the path {path} to saved model is not correct")

        state_dict = torch.load(path, map_location=torch.device(f'cuda:{self.gpu_ids[0]}') if torch.cuda.is_available() and (not self.no_gpu) else torch.device('cpu'))
        is_saved_distributed = all(t.startswith("module.") for t in state_dict["state_dict"])
        if not self.distributed and is_saved_distributed:
            state_dict["state_dict"] = {k[7:]: v for k,v in state_dict["state_dict"].items()}
        if self.distributed and not is_saved_distributed:
            state_dict["state_dict"] = {f"module.{k}": v for k,v in state_dict["state_dict"].items()}
        if load_params:
            state_dict["state_dict"] = {k:v for k,v in state_dict["state_dict"].items() if k in load_params}

        model.load_state_dict(state_dict=state_dict["state_dict"], strict=strict)
        if load_iter:
            self.epoch = state_dict["iter"] - 1
        if optimizer:
            optimizer.load_state_dict(state_dict=state_dict["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(state_dict=state_dict["scheduler_state_dict"])
        if 'rng_state' in state_dict:
            torch.set_rng_state(torch.ByteTensor(state_dict['rng_state'].cpu()))
        if 'cuda_rng_state' in state_dict:
            torch.cuda.set_rng_state(torch.ByteTensor(state_dict['cuda_rng_state'].cpu()))
        return None

    def get_misc(self):
        label_info_file = os.path.join(self.root, "label_info.json")
        train_file = os.path.join(self.root, "train.jsonl")
        dev_file = os.path.join(self.root, "dev.jsonl")
        test_file = os.path.join(self.root, "test.jsonl") if not self.add_test else os.path.join(self.root, "test.add.jsonl")
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
        with open(train_file, "rt") as f:
            train = [json.loads(line) for line in f]
        with open(dev_file, "rt") as f:
            dev = _to_instance([json.loads(line) for line in f], "dev")
        with open(test_file, "rt") as f:
            test = _to_instance([json.loads(line) for line in f], "test")

        epoch_steps = (len(train) - 1) // (self.batch_size * self.accumulation_steps * len(self.gpu_ids)) + 1

        if "roberta" in self.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=self.word_level)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dev = [tokenizer(
                text=t.tokens,
                is_split_into_words=isinstance(t.tokens, list),
                add_special_tokens=True,
                padding='longest') for t in dev]
        test = [tokenizer(
                text=t.tokens,
                is_split_into_words=isinstance(t.tokens, list),
                add_special_tokens=True,
                padding='longest') for t in test]

        return {
            "epoch_steps": epoch_steps,
            "dev_encodings": dev,
            "test_encodings": test,
            "label2id": label2id
        }


    def get_datasets(self, test_only=False) -> Tuple[Union[IDataset, None], Union[IDataset, None], IDataset]:

        label_info_file = os.path.join(self.root, "label_info.json")
        train_file = os.path.join(self.root, "train.jsonl")
        dev_file = os.path.join(self.root, "dev.jsonl")
        test_file = os.path.join(self.root, "test.jsonl") if not self.add_test else os.path.join(self.root, "test.add.jsonl")

        load_train_dev_file = not test_only
        build_train_dev_dataset = not test_only

        label_info = label2id = None
        train = dev = test = None

        train_dataset = None
        dev_dataset = None

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
                train = _to_instance([json.loads(line) for line in f], "train")
            with open(dev_file, "rt") as f:
                dev = _to_instance([json.loads(line) for line in f], "dev")


        with open(test_file, "rt") as f:
            test = _to_instance([json.loads(line) for line in f], "test")
        if "roberta" in self.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=self.word_level)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if build_train_dev_dataset:
            train_dataset = IDataset(
                instances=train,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=self.max_length,
                root=self.root if self.run_method in ["surrogate", "surrogate_distill"] else None)
            dev_dataset = IDataset(
                instances=dev,
                label2id=label2id,
                tokenizer=tokenizer,
                max_length=self.max_length,
                root=self.root if self.run_method in ["surrogate", "surrogate_distill"] else None)
        test_dataset = IDataset(
            instances=test,
            label2id=label2id,
            tokenizer=tokenizer,
            max_length=self.max_length,
            root=self.root if self.run_method in ["surrogate", "surrogate_distill"] else None)
        return {
            "train": train_dataset,
            "dev": dev_dataset,
            "test": test_dataset
        }

    def get_dataloaders(self,
        train_dataset,
        dev_dataset,
        test_dataset,
        collate_fn:Optional[Callable]=None,
        eval_collate_fn:Optional[Callable]=None,
        seed:Optional[int]=None,
        ):

        seed = int(time.time()) if seed is None else seed
        loaders = {}
        loaders["train"] = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=getattr(train_dataset, "collate_fn", None) if collate_fn is None else collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(seed)
        ) if train_dataset is not None else None
        loaders["dev"] = DataLoader(
            dataset=dev_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=getattr(dev_dataset, "collate_fn", None) if eval_collate_fn is None else eval_collate_fn,
            pin_memory=True,
            num_workers=self.num_workers
        ) if dev_dataset is not None else None
        loaders["test"] = DataLoader(
            dataset=test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=getattr(test_dataset, "collate_fn", None) if eval_collate_fn is None else eval_collate_fn,
            pin_memory=True,
            num_workers=self.num_workers)
        return loaders


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def worker_fn(rank, world_size, opts, net_module, net_kwargs, output_fn=None, output_kwargs={}, seeds=[42, 170, 682, 2730]):
    setup(rank, world_size)
    try:
        worker = Worker(opts)
        worker.set_model_params(net_module, **net_kwargs)
        worker.set_output_params(output_fn, **output_kwargs)
        model, output_fn, optimizer, scheduler, loaders, misc = worker.initialize(rank, seed=seeds[rank])

        if "roberta" in worker.model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(worker.model_name, add_prefix_space=worker.word_level)
        else:
            tokenizer = AutoTokenizer.from_pretrained(worker.model_name)

        eval_metric = F1MetricTag(-100, [0], misc["label2id"]['event'], tokenizer, save_dir=opts.log_dir, fix_span=False)
        pbar = tqdm(desc="Epoch 0", total=worker.train_step, ncols=100)

        worker.train(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            output_fn=output_fn,
            eval_loader=[loaders["dev"], loaders["test"]],
            eval_metric=eval_metric,
            eval_args=[misc["dev_encodings"], misc["test_encodings"]],
            progress=pbar,
            rank=rank)
    except Exception as e:
        cleanup()
        raise(e)
    cleanup()
