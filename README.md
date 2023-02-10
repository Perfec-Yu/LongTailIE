# Long-tail Learning

# Data

- We provide `data/maven` as an example data folder for the MAVEN dataset. Most baselines only needs `train/dev/test.jsonl` and `label_info.json` to run. Our models requires some additional preprocessing to improve the efficiency (see details below). 

# Running baselines

- We use MAVEN dataset as an example. For ACE, change `--root data/ace`, `--log-dir log_maven/{method}`, `--dataset ace` and `-n-class 34`. For FewNERD, change `--root data/fewnerd`, `--log-dir log_fewnerd/{method}`, `--dataset fewnerd`, `-n-class 69` and `--task-of-label entity`, and also add `--word-level`. Besides, CRT and LWS training is dependent on the Vanilla checkpoint. So please train the vanilla model first.
  
- Vanilla: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_maven/roberta-large-vanilla/ --eval-method macro --run-method vanilla --max-length 256 --dataset maven --min-epoch -1 --root data/maven/ --n-class 169 --task-of-label trigger`

- Momentum: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir /shared/nas/data/m1/pengfei4/research/imbalance_learning/log_maven/roberta-large-momentum/ --eval-method macro --run-method momentum --max-length 256 --dataset maven --min-epoch -1 --root /shared/nas/data/m1/pengfei4/research/imbalance_learning/data/maven/ --n-class 169 --task-of-label trigger`

- LWS: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_maven/roberta-large-lws/ --eval-method macro --run-method lws --max-length 256 --dataset maven --min-epoch -1 --root data/maven/ --lws-load-dir log_maven/roberta-large-vanilla --n-class 169 --task-of-label trigger`


- Focal: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_maven/roberta-large-focal/ --eval-method macro --run-method focal --max-length 256 --dataset maven --min-epoch -1 --root data/maven/ --n-class 169 --task-of-label trigger`


- CRT: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_maven/roberta-large-crt/ --eval-method macro --run-method crt --max-length 256 --dataset maven --min-epoch -1 --root data/maven/ --crt-load-dir log_maven/roberta-large-vanilla --n-class 169 --task-of-label trigger`

- Tau-Norm: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_maven/roberta-large-vanilla/ --eval-method macro --run-method tau_norm --tau-norm 0.8 --max-length 256 --dataset maven --min-epoch -1 --root data/maven/ --n-class 169 --task-of-label trigger`


- NCM: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_maven/roberta-large-vanilla/ --eval-method macro --run-method ncm --ncm-threshold 0.8 --max-length 256 --dataset maven --min-epoch -1 --root data/maven/ --n-class 169 --task-of-label trigger`

# Running Our Model
- Due to the two-stage training, we basically requires a vanilla training step before running the below command. The `--surrogate-load-dir log_maven/roberta-large-vanilla/` argument makes sure the script loads the saved checkpoint in the vanilla training stage.
- Besides, some features needs to be prepared as a pre-processing step to improve the training speed. `data/maven/contextual_features/` includes an example file for contextual features of a single training instances. It stores a tensor of size `seq_len x hidden_dim` which is contextual features (by masking each token individually, encoding the whole sentence with a pretrained-LM, and then extract the representation of the masked position). `data/maven/token_freq_mat.th` and `data/maven/weighted_type_tokens.th` are two tensors storing token frequencies and token frequencies for each type separately. Tokens are indexed with the token_id in the corresponding LM's tokenizer. Please load the provided example tensors for the MAVEN dataset to get the exact format.
- Ours: `CUDA_VISIBLE_DEVICES={GPU_ID}, python run_train.py --gpu 0 --model-name roberta-large --log-dir log_maven/roberta-large-surrogatedistilllayermod/ --eval-method macro --run-method surrogate_distill --max-length 256 --dataset maven --min-epoch -1 --root data/maven/ --n-class 169 --task-of-label trigger --surrogate-load-dir log_maven/roberta-large-vanilla/`

# Evaluation

For all evaluations, add `--test-only` to the above commands to run on the test data.