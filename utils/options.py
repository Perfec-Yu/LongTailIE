import argparse
import importlib
import os
import glob


def define_arguments(parser):
    parser.add_argument('--root', type=str, default="data/ace", help="")
    parser.add_argument('--batch-size', type=int, default=8, help="")
    parser.add_argument('--eval-batch-size', type=int, default=32, help="")
    parser.add_argument('--patience', type=int, default=5, help="")
    parser.add_argument('--n-momentum-heads', type=int, default=2, help="")
    parser.add_argument('--ncm-threshold', type=float, default=0., help="")
    parser.add_argument('--momentum-norm-factor', type=float, default=1./32, help="")
    parser.add_argument('--momentum-weight', type=float, default=1.5, help="")
    parser.add_argument('--surrogate-mu', type=float, default=0.5, help="")
    parser.add_argument('--surrogate-lam', type=float, default=0.1, help="")
    parser.add_argument('--mu', type=float, default=0.9998, help="")
    parser.add_argument('--tau-norm', type=float, default=1, help="")
    parser.add_argument('--lam', type=float, default=1e-4, help="")
    parser.add_argument('--focal-alpha', type=float, default=0.25, help="")
    parser.add_argument('--focal-gamma', type=float, default=2, help="")
    parser.add_argument('--kernel-size', type=int, default=256, help="")
    parser.add_argument('--n-class', type=int, default=34, help="")
    parser.add_argument('--n-dilations', type=int, default=3, help="")
    parser.add_argument('--hidden-dim', type=int, default=1024, help="")
    parser.add_argument('--max-length', type=int, default=256, help="")
    parser.add_argument('--task-of-label', choices=['trigger', 'entity'], default='trigger', help="")
    parser.add_argument('--dataset', choices=['ace', 'maven', 'fewnerd'], help="")
    parser.add_argument('--run-method', choices=['vanilla', 'focal', 'momentum', 'tau_norm', 'crt', 'lws', 'surrogate', 'ncm', 'surrogate_distill'], help="")
    parser.add_argument('--eval-method', choices=['micro', 'macro', 'type'], default='marco', help="")
    parser.add_argument('--accumulation-steps', type=int, default=1, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--gpu', type=str, default='1', help="gpu")
    parser.add_argument('--max-grad-norm', type=float, default=1, help="")
    parser.add_argument('--learning-rate', type=float, default=1e-5, help="")
    parser.add_argument('--weight-decay', type=float, default=1e-2, help="")
    parser.add_argument('--warmup', type=float, default=0, help="")
    parser.add_argument('--seed', type=str, default='42', help="random seed")
    parser.add_argument('--lws-load-dir', type=str, default="log_ace/roberta-large-vanilla", help="path to save log file")
    parser.add_argument('--crt-load-dir', type=str, default="log_ace/roberta-large-vanilla", help="path to save log file")
    parser.add_argument('--surrogate-load-dir', type=str, default="log_fewnerd/roberta-large-vanilla", help="path to save log file")
    parser.add_argument('--log-dir', type=str, default="log_ace/roberta-large", help="path to save log file")
    parser.add_argument('--model-name', type=str, default="roberta-large", help="pretrained lm name")
    parser.add_argument('--num-workers', type=int, default=0, help='epochs to train')
    parser.add_argument('--surrogate-fusion-layer', type=int, default=0, help='epochs to train')
    parser.add_argument('--train-epoch', type=int, default=50, help='epochs to train')
    parser.add_argument('--min-epoch', type=int, default=20, help='epochs to train')
    parser.add_argument('--train-step', type=int, default=-1, help='steps to train')
    parser.add_argument('--use-crf', action="store_true", help='')
    parser.add_argument('--word-level', action="store_true", help='')
    parser.add_argument('--test-only', action="store_true", help='')
    parser.add_argument('--continue-train', action="store_true", help='continue training')
    parser.add_argument('--use-original-lm', action="store_true", help='')
    parser.add_argument('--surrogate-na', action="store_true", help='')
    parser.add_argument('--crt-leave-na', action="store_true", help='')
    parser.add_argument('--lws-bias', action="store_true", help='')
    parser.add_argument('--tau-norm-bias', action="store_true", help='')
    parser.add_argument('--add-test', action="store_true", help='')
    parser.add_argument('--get-history', action="store_true", help='')
    parser.add_argument('--dev-only', action="store_true", help='')
    parser.add_argument('--surrogate-lws', action="store_true", help='')
    parser.add_argument('--surrogate-no-att-loss', action="store_true", help='')
    parser.add_argument('--clean-log-dir', action="store_true", help='')


def parse_arguments(no_clean_dir:bool=False):
    parser = argparse.ArgumentParser()
    cwd = os.getcwd()
    path = "default_options" if cwd.endswith("utils") else "utils.default_options"
    default_options = importlib.util.find_spec(path)
    if default_options:
        define_default_arguments = default_options.loader.load_module().define_default_arguments
        define_default_arguments(parser)
    else:
        define_arguments(parser)
    args = parser.parse_args()
    if not no_clean_dir and args.clean_log_dir and (not args.test_only) and (not args.continue_train) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            os.remove(_t)

    return args
