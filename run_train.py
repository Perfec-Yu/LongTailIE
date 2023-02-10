import os
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from models.seq import SeqCls

from utils.options import parse_arguments
from utils.utils import F1MetricTag, F1Record, MacroF1Record, MixF1Record, get_weight_tensor, get_maven_candidates, get_freq_tensor, get_type_token_tensor
from utils.worker import Worker, worker_fn


import torch.multiprocessing as mp


def main():
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    opts = parse_arguments(no_clean_dir=True)
    SEEDS = [int(t) for t in opts.seed.split(",")]

    net_kwargs = {
        "nclass": opts.n_class,
        "model_name": opts.model_name,
        "use_crf": opts.use_crf,
        "task_of_label": opts.task_of_label
    }
    if opts.run_method == 'focal' or opts.run_method == 'backdoor+focal':
        net_kwargs.update({"focal": True, "alpha": opts.focal_alpha, "gamma": opts.focal_gamma})
    if opts.run_method == 'backdoor' or opts.run_method == 'backdoor+focal':
        net_kwargs.update({"featurewise": True, "kernel_size": opts.kernel_size, "n_dilations": opts.n_dilations, "lam": opts.lam})
    if opts.run_method == 'momentum':
        net_kwargs.update({"momentum": True, "n_momentum_heads": opts.n_momentum_heads, "momentum_norm_factor": opts.momentum_norm_factor, "momentum_weight": opts.momentum_weight, "mu": opts.mu})
    if opts.run_method == 'tau_norm':
        net_kwargs.update({"tau_norm":True, "tau": opts.tau_norm, "tau_norm_bias": opts.tau_norm_bias})
    if opts.run_method == 'crt':
        net_kwargs.update({"crt":True, "class_balance_tensor": get_weight_tensor(opts.dataset, leave_na=opts.crt_leave_na)})
    if opts.run_method == 'lws':
        net_kwargs.update({"lws":True, "lws_bias": opts.lws_bias, "class_balance_tensor": get_weight_tensor(opts.dataset, leave_na=opts.crt_leave_na)})
    if opts.run_method == 'surrogate':
        net_kwargs.update({"surrogate":True, "surrogate_mu": opts.surrogate_mu, "surrogate_lam": opts.surrogate_lam, "surrogate_na": opts.surrogate_na})
    if opts.run_method == 'surrogate_distill':
        net_kwargs.update({"surrogate_distill":True, "surrogate_mu": opts.surrogate_mu, "surrogate_lam": opts.surrogate_lam, "surrogate_att_loss": not opts.surrogate_no_att_loss, 'surrogate_fusion_layer': opts.surrogate_fusion_layer, "token_freq_tensor": get_freq_tensor(opts.root), "type_token_tensor": get_type_token_tensor(opts.root)})
    if opts.surrogate_lws:
        net_kwargs.update({"surrogate_lws":True, "lws_bias": opts.lws_bias, "class_balance_tensor": get_weight_tensor(opts.dataset, leave_na=opts.crt_leave_na)})
    if opts.run_method == 'ncm':
        net_kwargs.update({"ncm":True, "ncm_th": opts.ncm_threshold})
    gpu = opts.gpu
    gpu_ids = [int(t.strip()) for t in opts.gpu.strip().split(",")]
    world_size = len(gpu_ids)

    if len(gpu_ids) > 1:
        mp.spawn(worker_fn,
                args=(world_size, opts, SeqCls, net_kwargs),
                nprocs=world_size,
                join=True)
    else:
        results = []
        for seed in SEEDS:
            worker = Worker(opts)
            worker.set_model_params(SeqCls, **net_kwargs)
            model, output_fn, optimizer, scheduler, loaders, misc = worker.initialize(-1, seed=seed)

            if opts.get_history:
                if os.path.exists(os.path.join(opts.log_dir, f"model.{seed}")):
                    worker.load(model, path=os.path.join(opts.log_dir, f"model.{seed}"))
                    history = worker.get_history(model, loaders["train"], feat_func=lambda model:{"history_input": model.history_input.detach().cpu(), "history_count": model.history_count.detach().cpu()})
                    torch.save(history, os.path.join(opts.log_dir, f"history.{seed}"))
                continue

            if opts.run_method == 'crt' and not opts.test_only and not opts.use_original_lm:
                worker.load(model, path=os.path.join(opts.crt_load_dir, f"model.{seed}"), strict=False, load_iter=False, load_params={n for n, p in model.named_parameters() if 'pretrained_lm' in n})
                print(len({n for n, p in model.named_parameters() if 'pretrained_lm' in n}))
            if (opts.run_method == 'lws' or opts.surrogate_lws) and not opts.test_only and not opts.use_original_lm:
                worker.load(model, path=os.path.join(opts.lws_load_dir, f"model.{seed}"), strict=False, load_iter=False)
            if opts.run_method == 'surrogate_distill' and not opts.test_only:
                worker.load(model, path=os.path.join(opts.surrogate_load_dir, f"model.{seed}"), strict=False, load_iter=False)

            if "roberta" in opts.model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(opts.model_name, add_prefix_space=opts.word_level)
            else:
                tokenizer = AutoTokenizer.from_pretrained(opts.model_name)
            candidates = None if opts.dataset != "maven" else get_maven_candidates(opts.root, opts.model_name)
            f1metric = F1MetricTag(-100, [0], misc["label2id"]['event' if opts.task_of_label == 'trigger' else 'entity'], tokenizer, save_dir=opts.log_dir, save_name=f"prediction_{seed}.jsonl", save_annotation=opts.test_only,fix_span='none', word_level=opts.word_level, candidates=candidates)
            eval_metric = lambda x,y:f1metric(x,y)[0 if opts.eval_method == "micro" else 2 if opts.eval_method == 'macro' else 1]
            if opts.eval_method == 'type':
                assert opts.test_only

            if opts.test_only:
                if os.path.exists(os.path.join(opts.log_dir, f"model.{seed}")):
                    worker.load(model, path=os.path.join(opts.log_dir, f"model.{seed}"))
                    if opts.run_method == "surrogate":
                        history = torch.load(os.path.join(opts.log_dir, f"history.{seed}"))
                        model.surrogate_cls.history_input = history["history_input"].to(model.surrogate_cls.history_input.device)
                        model.surrogate_cls.history_count[:] = 1
                    if opts.run_method == "ncm":
                        history = torch.load(os.path.join(opts.log_dir, f"history.{seed}"))
                        model.history_input = history["history_input"].to(model.pretrained_lm.device)
                    results.append(worker.eval(
                        model=model,
                        loader=loaders["test"],
                        eval_metric=eval_metric,
                        eval_args=[misc["test_encodings"]]
                    )[0])
            else:
                eval_loader =[loaders["dev"]] if opts.dev_only else [loaders["dev"], loaders['test']]
                eval_args = [misc['dev_encodings']] if opts.dev_only else [misc["dev_encodings"], misc["test_encodings"]]
                worker.train(
                    model=model,
                    loader=loaders["train"],
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_fn=output_fn,
                    eval_loader=eval_loader,
                    eval_metric=eval_metric,
                    eval_args=eval_args,
                    save_postfix=str(seed)
                )
            del worker, model, output_fn, optimizer, scheduler, loaders, misc, eval_metric
            torch.cuda.empty_cache()
    if opts.test_only:
        if isinstance(results[0], F1Record) or isinstance(results[0], MacroF1Record) or isinstance(results[0], MixF1Record):
            results = [r.named_result for r in results]
        average = {k: 0. for k in results[0]}
        for i, r in enumerate(results):
            for k, v in r.items():
                if isinstance(v, F1Record):
                    v = np.array(v.full_result)
                if i == 0:
                    average[k] = v
                else:
                    average[k] += v
        average = {k: v / len(results) for k,v in average.items()}
        print(average)
        if opts.eval_method == 'type':
            if opts.run_method in ["tau_norm", "ncm"]:
                torch.save({"full_results": results, "average_results": average}, os.path.join(opts.log_dir, f"scores_by_type_{opts.run_method}.th"))
            else:
                print('saved')
                torch.save({"full_results": results, "average_results": average}, os.path.join(opts.log_dir, "scores_by_type.th"))

if __name__ == "__main__":
    main()
