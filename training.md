(LinOSS) khanhnt@aiotlab-server:/project/khanhnt/control_theory/llm-jepa$ python compare_three_method.py   --dataset_name synth --data_prefix datasets/   --num_epochs 1 --nproc 2 --batch_size 2 --grad_accum 4   --only_dynamics --tube_log_interval 50
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1776022022.207599  477234 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1776022022.285162  477234 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX512_FP16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1776022024.617019  477234 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

>>> rendezvous 127.0.0.1:40659 (set MASTER_ADDR/MASTER_PORT to override)
>>> torchrun --nproc_per_node=2 --master-addr=127.0.0.1 --master-port=40659 /project/khanhnt/control_theory/llm-jepa/stp.py --train_file /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl --output_dir /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics --num_epochs 1 --finetune_seed 42 --model_name meta-llama/Llama-3.2-1B-Instruct --learning_rate 2e-05 --batch_size 2 --grad_accum 4 --max_length 512 --last_token -2 --lbd 0.01 --predictors 0 --dynamics_tube --tube_gamma 0.95 --tube_tau 0.001 --lbd_ts 0.0 --tube_log_interval 50
W0412 19:27:06.901000 477539 torch/distributed/run.py:766] 
W0412 19:27:06.901000 477539 torch/distributed/run.py:766] *****************************************
W0412 19:27:06.901000 477539 torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0412 19:27:06.901000 477539 torch/distributed/run.py:766] *****************************************
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=1
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=0

1. Loading model and tokenizer...
Added 11 new special tokens
Added <|mask|> token: 128267
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`

2. Loading and preparing dataset...
Loading training data from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
Loaded 8000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No evaluation file provided
Train samples: 8000
No evaluation dataset

3. Initializing representation trainer...
/project/khanhnt/control_theory/llm-jepa/stp.py:696: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `RepresentationTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
/project/khanhnt/control_theory/llm-jepa/stp.py:696: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `RepresentationTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.

4. Starting training...
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
  0%|                                                                             | 0/500 [00:00<?, ?it/s][tube_diag] step=0 mean_V_curr=0.38428 mean_V_prev=0.38428 viol_rate=0.5385 tube_softplus_mean=0.70469 lm_loss=5.43424 jepa_total=0.70469
{'loss': 6.4359, 'grad_norm': 34.5, 'learning_rate': 1.9640000000000002e-05, 'epoch': 0.02}               
{'loss': 2.8012, 'grad_norm': 33.5, 'learning_rate': 1.9240000000000002e-05, 'epoch': 0.04}               
{'loss': 2.2672, 'grad_norm': 66.0, 'learning_rate': 1.884e-05, 'epoch': 0.06}                            
{'loss': 1.7091, 'grad_norm': 47.0, 'learning_rate': 1.8440000000000003e-05, 'epoch': 0.08}               
{'loss': 1.1696, 'grad_norm': 21.125, 'learning_rate': 1.8040000000000003e-05, 'epoch': 0.1}              
 10%|██████▊                                                             | 50/500 [01:17<12:10,  1.62s/it][tube_diag] step=50 mean_V_curr=0.39850 mean_V_prev=0.39850 viol_rate=0.6400 tube_softplus_mean=0.70624 lm_loss=0.32191 jepa_total=0.70624
{'loss': 1.0552, 'grad_norm': 19.0, 'learning_rate': 1.764e-05, 'epoch': 0.12}                            
{'loss': 1.0067, 'grad_norm': 17.5, 'learning_rate': 1.724e-05, 'epoch': 0.14}                            
{'loss': 0.973, 'grad_norm': 14.125, 'learning_rate': 1.684e-05, 'epoch': 0.16}                           
{'loss': 0.9694, 'grad_norm': 15.875, 'learning_rate': 1.6440000000000002e-05, 'epoch': 0.18}             
{'loss': 0.9064, 'grad_norm': 15.1875, 'learning_rate': 1.6040000000000002e-05, 'epoch': 0.2}             
 20%|█████████████▍                                                     | 100/500 [02:34<09:03,  1.36s/it][tube_diag] step=100 mean_V_curr=0.39081 mean_V_prev=0.39081 viol_rate=0.6667 tube_softplus_mean=0.70505 lm_loss=0.28167 jepa_total=0.70505
{'loss': 0.9064, 'grad_norm': 11.6875, 'learning_rate': 1.5640000000000003e-05, 'epoch': 0.22}            
{'loss': 0.8638, 'grad_norm': 11.25, 'learning_rate': 1.5240000000000001e-05, 'epoch': 0.24}              
{'loss': 0.8635, 'grad_norm': 9.6875, 'learning_rate': 1.4840000000000002e-05, 'epoch': 0.26}             
{'loss': 0.8414, 'grad_norm': 12.0625, 'learning_rate': 1.444e-05, 'epoch': 0.28}                         
{'loss': 0.8281, 'grad_norm': 9.25, 'learning_rate': 1.4040000000000001e-05, 'epoch': 0.3}                
 30%|████████████████████                                               | 150/500 [03:53<10:07,  1.73s/it][tube_diag] step=150 mean_V_curr=0.36847 mean_V_prev=0.36847 viol_rate=0.5652 tube_softplus_mean=0.70527 lm_loss=0.19305 jepa_total=0.70527
{'loss': 0.8379, 'grad_norm': 16.125, 'learning_rate': 1.3640000000000002e-05, 'epoch': 0.32}             
{'loss': 0.8188, 'grad_norm': 18.375, 'learning_rate': 1.3240000000000002e-05, 'epoch': 0.34}             
{'loss': 0.8129, 'grad_norm': 11.6875, 'learning_rate': 1.284e-05, 'epoch': 0.36}                         
{'loss': 0.797, 'grad_norm': 10.125, 'learning_rate': 1.2440000000000001e-05, 'epoch': 0.38}              
{'loss': 0.7959, 'grad_norm': 11.0, 'learning_rate': 1.204e-05, 'epoch': 0.4}                             
 40%|██████████████████████████▊                                        | 200/500 [05:12<08:11,  1.64s/it][tube_diag] step=200 mean_V_curr=0.33969 mean_V_prev=0.33969 viol_rate=0.6230 tube_softplus_mean=0.70273 lm_loss=0.17084 jepa_total=0.70273
{'loss': 0.7918, 'grad_norm': 7.28125, 'learning_rate': 1.164e-05, 'epoch': 0.42}                         
{'loss': 0.8121, 'grad_norm': 12.125, 'learning_rate': 1.1240000000000002e-05, 'epoch': 0.44}             
{'loss': 0.7972, 'grad_norm': 8.9375, 'learning_rate': 1.0840000000000001e-05, 'epoch': 0.46}             
{'loss': 0.7806, 'grad_norm': 9.375, 'learning_rate': 1.0440000000000002e-05, 'epoch': 0.48}              
{'loss': 0.7828, 'grad_norm': 11.5, 'learning_rate': 1.004e-05, 'epoch': 0.5}                             
 50%|█████████████████████████████████▌                                 | 250/500 [06:29<05:39,  1.36s/it][tube_diag] step=250 mean_V_curr=0.35469 mean_V_prev=0.35469 viol_rate=0.6364 tube_softplus_mean=0.70329 lm_loss=0.16468 jepa_total=0.70329
{'loss': 0.7873, 'grad_norm': 10.8125, 'learning_rate': 9.640000000000001e-06, 'epoch': 0.52}             
{'loss': 0.7896, 'grad_norm': 8.9375, 'learning_rate': 9.240000000000001e-06, 'epoch': 0.54}              
{'loss': 0.7782, 'grad_norm': 7.53125, 'learning_rate': 8.84e-06, 'epoch': 0.56}                          
{'loss': 0.7626, 'grad_norm': 10.4375, 'learning_rate': 8.44e-06, 'epoch': 0.58}                          
{'loss': 0.781, 'grad_norm': 8.9375, 'learning_rate': 8.040000000000001e-06, 'epoch': 0.6}                
 60%|████████████████████████████████████████▏                          | 300/500 [07:47<05:55,  1.78s/it][tube_diag] step=300 mean_V_curr=0.31899 mean_V_prev=0.31899 viol_rate=0.5714 tube_softplus_mean=0.70257 lm_loss=0.24092 jepa_total=0.70257
{'loss': 0.7566, 'grad_norm': 7.78125, 'learning_rate': 7.640000000000001e-06, 'epoch': 0.62}             
{'loss': 0.7821, 'grad_norm': 12.625, 'learning_rate': 7.24e-06, 'epoch': 0.64}                           
{'loss': 0.7524, 'grad_norm': 8.6875, 'learning_rate': 6.8400000000000014e-06, 'epoch': 0.66}             
{'loss': 0.7624, 'grad_norm': 9.25, 'learning_rate': 6.440000000000001e-06, 'epoch': 0.68}                
{'loss': 0.7575, 'grad_norm': 8.875, 'learning_rate': 6.040000000000001e-06, 'epoch': 0.7}                
 70%|██████████████████████████████████████████████▉                    | 350/500 [09:12<04:14,  1.70s/it][tube_diag] step=350 mean_V_curr=0.35322 mean_V_prev=0.35322 viol_rate=0.7561 tube_softplus_mean=0.70395 lm_loss=0.21844 jepa_total=0.70395
{'loss': 0.7684, 'grad_norm': 10.375, 'learning_rate': 5.64e-06, 'epoch': 0.72}                           
{'loss': 0.7583, 'grad_norm': 9.8125, 'learning_rate': 5.240000000000001e-06, 'epoch': 0.74}              
{'loss': 0.7449, 'grad_norm': 9.9375, 'learning_rate': 4.84e-06, 'epoch': 0.76}                           
{'loss': 0.7715, 'grad_norm': 13.25, 'learning_rate': 4.440000000000001e-06, 'epoch': 0.78}               
{'loss': 0.7588, 'grad_norm': 8.9375, 'learning_rate': 4.04e-06, 'epoch': 0.8}                            
 80%|█████████████████████████████████████████████████████▌             | 400/500 [10:35<02:33,  1.54s/it][tube_diag] step=400 mean_V_curr=0.36050 mean_V_prev=0.36050 viol_rate=0.5882 tube_softplus_mean=0.70505 lm_loss=0.20031 jepa_total=0.70505
{'loss': 0.7641, 'grad_norm': 8.375, 'learning_rate': 3.6400000000000003e-06, 'epoch': 0.82}              
{'loss': 0.7577, 'grad_norm': 7.53125, 'learning_rate': 3.2400000000000003e-06, 'epoch': 0.84}            
{'loss': 0.7621, 'grad_norm': 9.875, 'learning_rate': 2.84e-06, 'epoch': 0.86}                            
{'loss': 0.7667, 'grad_norm': 9.5, 'learning_rate': 2.4400000000000004e-06, 'epoch': 0.88}                
{'loss': 0.7421, 'grad_norm': 7.90625, 'learning_rate': 2.04e-06, 'epoch': 0.9}                           
 90%|████████████████████████████████████████████████████████████▎      | 450/500 [11:54<01:21,  1.64s/it][tube_diag] step=450 mean_V_curr=0.39630 mean_V_prev=0.39630 viol_rate=0.6250 tube_softplus_mean=0.70673 lm_loss=0.21478 jepa_total=0.70673
{'loss': 0.7638, 'grad_norm': 8.3125, 'learning_rate': 1.6400000000000002e-06, 'epoch': 0.92}             
{'loss': 0.7302, 'grad_norm': 9.375, 'learning_rate': 1.2400000000000002e-06, 'epoch': 0.94}              
{'loss': 0.7445, 'grad_norm': 9.125, 'learning_rate': 8.400000000000001e-07, 'epoch': 0.96}               
{'loss': 0.7494, 'grad_norm': 6.0, 'learning_rate': 4.4e-07, 'epoch': 0.98}                               
{'loss': 0.7537, 'grad_norm': 9.6875, 'learning_rate': 4e-08, 'epoch': 1.0}                               
{'train_runtime': 792.3807, 'train_samples_per_second': 10.096, 'train_steps_per_second': 0.631, 'train_loss': 1.0133952159881592, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████| 500/500 [13:12<00:00,  1.58s/it]

5. Saving final model...

✅ Training completed! Model saved to /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics

🎉 Fine-tuning finished successfully!
[rank0]:[W412 19:41:01.342662232 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
Loaded 2000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_test.jsonl
PPL[dynamics]: 100%|████████████████████████████████████████████████████| 500/500 [01:04<00:00,  7.71it/s]

========== Perplexity comparison ==========
  dynamics      PPL = 1.19731016879851
