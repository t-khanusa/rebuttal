(LinOSS) khanhnt@aiotlab-server:/project/khanhnt/control_theory/llm-jepa$ python compare_three_method.py --dataset_name synth --data_prefix datasets/ --num_epochs 1 --nproc 2 --batch_size 2 --grad_accum 4
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1776011594.095963  193036 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1776011594.157303  193036 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX512_FP16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1776011596.434675  193036 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

>>> rendezvous 127.0.0.1:29501 (set MASTER_ADDR/MASTER_PORT to override)
>>> torchrun --nproc_per_node=2 --master-addr=127.0.0.1 --master-port=29501 /project/khanhnt/control_theory/llm-jepa/stp.py --train_file /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl --output_dir /project/khanhnt/control_theory/llm-jepa/compare_three_runs/regular --num_epochs 1 --finetune_seed 42 --model_name meta-llama/Llama-3.2-1B-Instruct --learning_rate 2e-05 --batch_size 2 --grad_accum 4 --max_length 512 --last_token -2 --lbd 0.02 --predictors 0 --regular
W0412 16:33:18.353000 195060 torch/distributed/run.py:766] 
W0412 16:33:18.353000 195060 torch/distributed/run.py:766] *****************************************
W0412 16:33:18.353000 195060 torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0412 16:33:18.353000 195060 torch/distributed/run.py:766] *****************************************
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/regular
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=0

1. Loading model and tokenizer...
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/regular
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=1
tokenizer_config.json: 54.5kB [00:00, 54.0MB/s]
tokenizer.json: 9.09MB [00:01, 7.11MB/s]
special_tokens_map.json: 100%|███████████████████████████████████████████████████████████████| 296/296 [00:00<00:00, 1.02MB/s]
Added 11 new special tokens
Added <|mask|> token: 128267
config.json: 100%|███████████████████████████████████████████████████████████████████████████| 877/877 [00:00<00:00, 3.09MB/s]
model.safetensors: 100%|█████████████████████████████████████████████████████████████████| 2.47G/2.47G [00:24<00:00, 99.5MB/s]
generation_config.json: 100%|█████████████████████████████████████████████████████████████████| 189/189 [00:00<00:00, 535kB/s]
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`

2. Loading and preparing dataset...
Loading training data from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
Generating train split: 8000 examples [00:00, 498802.32 examples/s]
Map:   0%|                                                                                    | 0/8000 [00:00<?, ? examples/s]Loaded 8000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
Map: 100%|████████████████████████████████████████████████████████████████████████| 8000/8000 [00:50<00:00, 157.09 examples/s]
/project/khanhnt/control_theory/llm-jepa/stp.py:1626: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(
Map: 100%|████████████████████████████████████████████████████████████████████████| 8000/8000 [00:50<00:00, 156.89 examples/s]
No evaluation file provided
Train samples: 8000
No evaluation dataset

3. Initializing regular trainer...
/project/khanhnt/control_theory/llm-jepa/stp.py:1626: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  trainer = Trainer(

4. Starting training...
{'loss': 1.579, 'grad_norm': 8.375, 'learning_rate': 1.9640000000000002e-05, 'epoch': 0.02}                                   
{'loss': 0.6825, 'grad_norm': 8.1875, 'learning_rate': 1.9240000000000002e-05, 'epoch': 0.04}                                 
{'loss': 0.5499, 'grad_norm': 16.25, 'learning_rate': 1.884e-05, 'epoch': 0.06}                                               
{'loss': 0.4127, 'grad_norm': 12.0625, 'learning_rate': 1.8440000000000003e-05, 'epoch': 0.08}                                
{'loss': 0.2807, 'grad_norm': 5.3125, 'learning_rate': 1.8040000000000003e-05, 'epoch': 0.1}                                  
{'loss': 0.2525, 'grad_norm': 4.75, 'learning_rate': 1.764e-05, 'epoch': 0.12}                                                
{'loss': 0.2397, 'grad_norm': 3.6875, 'learning_rate': 1.724e-05, 'epoch': 0.14}                                              
{'loss': 0.2329, 'grad_norm': 3.390625, 'learning_rate': 1.684e-05, 'epoch': 0.16}                                            
{'loss': 0.2321, 'grad_norm': 3.90625, 'learning_rate': 1.6440000000000002e-05, 'epoch': 0.18}                                
{'loss': 0.2158, 'grad_norm': 3.625, 'learning_rate': 1.6040000000000002e-05, 'epoch': 0.2}                                   
{'loss': 0.2157, 'grad_norm': 2.859375, 'learning_rate': 1.5640000000000003e-05, 'epoch': 0.22}                               
{'loss': 0.2065, 'grad_norm': 2.796875, 'learning_rate': 1.5240000000000001e-05, 'epoch': 0.24}                               
{'loss': 0.2058, 'grad_norm': 2.640625, 'learning_rate': 1.4840000000000002e-05, 'epoch': 0.26}                               
{'loss': 0.1997, 'grad_norm': 2.921875, 'learning_rate': 1.444e-05, 'epoch': 0.28}                                            
{'loss': 0.1965, 'grad_norm': 2.40625, 'learning_rate': 1.4040000000000001e-05, 'epoch': 0.3}                                 
{'loss': 0.1976, 'grad_norm': 3.125, 'learning_rate': 1.3640000000000002e-05, 'epoch': 0.32}                                  
{'loss': 0.1933, 'grad_norm': 2.71875, 'learning_rate': 1.3240000000000002e-05, 'epoch': 0.34}                                
{'loss': 0.1931, 'grad_norm': 2.890625, 'learning_rate': 1.284e-05, 'epoch': 0.36}                                            
{'loss': 0.1895, 'grad_norm': 2.359375, 'learning_rate': 1.2440000000000001e-05, 'epoch': 0.38}                               
{'loss': 0.1889, 'grad_norm': 2.75, 'learning_rate': 1.204e-05, 'epoch': 0.4}                                                 
{'loss': 0.1888, 'grad_norm': 1.8671875, 'learning_rate': 1.164e-05, 'epoch': 0.42}                                           
{'loss': 0.193, 'grad_norm': 2.921875, 'learning_rate': 1.1240000000000002e-05, 'epoch': 0.44}                                
{'loss': 0.1905, 'grad_norm': 2.234375, 'learning_rate': 1.0840000000000001e-05, 'epoch': 0.46}                               
{'loss': 0.1856, 'grad_norm': 2.609375, 'learning_rate': 1.0440000000000002e-05, 'epoch': 0.48}                               
{'loss': 0.1866, 'grad_norm': 3.421875, 'learning_rate': 1.004e-05, 'epoch': 0.5}                                             
{'loss': 0.1864, 'grad_norm': 2.578125, 'learning_rate': 9.640000000000001e-06, 'epoch': 0.52}                                
{'loss': 0.1881, 'grad_norm': 3.09375, 'learning_rate': 9.240000000000001e-06, 'epoch': 0.54}                                 
{'loss': 0.1851, 'grad_norm': 1.78125, 'learning_rate': 8.84e-06, 'epoch': 0.56}                                              
{'loss': 0.1808, 'grad_norm': 2.46875, 'learning_rate': 8.44e-06, 'epoch': 0.58}                                              
{'loss': 0.1854, 'grad_norm': 2.1875, 'learning_rate': 8.040000000000001e-06, 'epoch': 0.6}                                   
{'loss': 0.179, 'grad_norm': 2.09375, 'learning_rate': 7.640000000000001e-06, 'epoch': 0.62}                                  
{'loss': 0.1865, 'grad_norm': 2.859375, 'learning_rate': 7.24e-06, 'epoch': 0.64}                                             
{'loss': 0.1795, 'grad_norm': 2.15625, 'learning_rate': 6.8400000000000014e-06, 'epoch': 0.66}                                
{'loss': 0.1805, 'grad_norm': 2.28125, 'learning_rate': 6.440000000000001e-06, 'epoch': 0.68}                                 
{'loss': 0.1797, 'grad_norm': 2.1875, 'learning_rate': 6.040000000000001e-06, 'epoch': 0.7}                                   
{'loss': 0.1821, 'grad_norm': 2.5625, 'learning_rate': 5.64e-06, 'epoch': 0.72}                                               
{'loss': 0.1792, 'grad_norm': 2.40625, 'learning_rate': 5.240000000000001e-06, 'epoch': 0.74}                                 
{'loss': 0.1766, 'grad_norm': 2.4375, 'learning_rate': 4.84e-06, 'epoch': 0.76}                                               
{'loss': 0.1827, 'grad_norm': 3.46875, 'learning_rate': 4.440000000000001e-06, 'epoch': 0.78}                                 
{'loss': 0.1802, 'grad_norm': 2.03125, 'learning_rate': 4.04e-06, 'epoch': 0.8}                                               
{'loss': 0.1811, 'grad_norm': 1.859375, 'learning_rate': 3.6400000000000003e-06, 'epoch': 0.82}                               
{'loss': 0.1795, 'grad_norm': 1.8828125, 'learning_rate': 3.2400000000000003e-06, 'epoch': 0.84}                              
{'loss': 0.1811, 'grad_norm': 2.46875, 'learning_rate': 2.84e-06, 'epoch': 0.86}                                              
{'loss': 0.1821, 'grad_norm': 2.34375, 'learning_rate': 2.4400000000000004e-06, 'epoch': 0.88}                                
{'loss': 0.1759, 'grad_norm': 2.03125, 'learning_rate': 2.04e-06, 'epoch': 0.9}                                               
{'loss': 0.1814, 'grad_norm': 2.0625, 'learning_rate': 1.6400000000000002e-06, 'epoch': 0.92}                                 
{'loss': 0.1724, 'grad_norm': 2.296875, 'learning_rate': 1.2400000000000002e-06, 'epoch': 0.94}                               
{'loss': 0.1763, 'grad_norm': 2.25, 'learning_rate': 8.400000000000001e-07, 'epoch': 0.96}                                    
{'loss': 0.1773, 'grad_norm': 1.46875, 'learning_rate': 4.4e-07, 'epoch': 0.98}                                               
{'loss': 0.179, 'grad_norm': 2.359375, 'learning_rate': 4e-08, 'epoch': 1.0}                                                  
{'train_runtime': 537.1694, 'train_samples_per_second': 14.893, 'train_steps_per_second': 0.931, 'train_loss': 0.2425364923477173, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████████████████████████| 500/500 [08:57<00:00,  1.07s/it]

5. Saving final model...

✅ Training completed! Model saved to /project/khanhnt/control_theory/llm-jepa/compare_three_runs/regular

🎉 Fine-tuning finished successfully!
[rank0]:[W412 16:44:11.574909322 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

>>> rendezvous 127.0.0.1:29501 (set MASTER_ADDR/MASTER_PORT to override)
>>> torchrun --nproc_per_node=2 --master-addr=127.0.0.1 --master-port=29501 /project/khanhnt/control_theory/llm-jepa/stp.py --train_file /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl --output_dir /project/khanhnt/control_theory/llm-jepa/compare_three_runs/stp --num_epochs 1 --finetune_seed 42 --model_name meta-llama/Llama-3.2-1B-Instruct --learning_rate 2e-05 --batch_size 2 --grad_accum 4 --max_length 512 --last_token -2 --lbd 0.02 --predictors 0
W0412 16:44:14.367000 273895 torch/distributed/run.py:766] 
W0412 16:44:14.367000 273895 torch/distributed/run.py:766] *****************************************
W0412 16:44:14.367000 273895 torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0412 16:44:14.367000 273895 torch/distributed/run.py:766] *****************************************
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/stp
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=0

1. Loading model and tokenizer...
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/stp
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=1
Added 11 new special tokens
Added <|mask|> token: 128267
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`

2. Loading and preparing dataset...
Loading training data from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
Loaded 8000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
Map: 100%|████████████████████████████████████████████████████████████████████████| 8000/8000 [00:40<00:00, 198.75 examples/s]
Map: 100%|████████████████████████████████████████████████████████████████████████| 8000/8000 [00:40<00:00, 201.40 examples/s]/project/khanhnt/control_theory/llm-jepa/stp.py:695: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `RepresentationTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Map: 100%|████████████████████████████████████████████████████████████████████████| 8000/8000 [00:40<00:00, 198.34 examples/s]
No evaluation file provided
Train samples: 8000
No evaluation dataset

3. Initializing representation trainer...
/project/khanhnt/control_theory/llm-jepa/stp.py:695: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `RepresentationTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.

4. Starting training...
{'loss': 6.4573, 'grad_norm': 34.25, 'learning_rate': 1.9640000000000002e-05, 'epoch': 0.02}                                  
{'loss': 2.8136, 'grad_norm': 33.25, 'learning_rate': 1.9240000000000002e-05, 'epoch': 0.04}                                  
{'loss': 2.2657, 'grad_norm': 64.5, 'learning_rate': 1.884e-05, 'epoch': 0.06}                                                
{'loss': 1.7045, 'grad_norm': 49.25, 'learning_rate': 1.8440000000000003e-05, 'epoch': 0.08}                                  
{'loss': 1.1505, 'grad_norm': 21.25, 'learning_rate': 1.8040000000000003e-05, 'epoch': 0.1}                                   
{'loss': 1.0386, 'grad_norm': 19.0, 'learning_rate': 1.764e-05, 'epoch': 0.12}                                                
{'loss': 0.9813, 'grad_norm': 18.25, 'learning_rate': 1.724e-05, 'epoch': 0.14}                                               
{'loss': 0.9479, 'grad_norm': 14.0, 'learning_rate': 1.684e-05, 'epoch': 0.16}                                                
{'loss': 0.9463, 'grad_norm': 15.125, 'learning_rate': 1.6440000000000002e-05, 'epoch': 0.18}                                 
{'loss': 0.8823, 'grad_norm': 14.4375, 'learning_rate': 1.6040000000000002e-05, 'epoch': 0.2}                                 
{'loss': 0.8784, 'grad_norm': 10.875, 'learning_rate': 1.5640000000000003e-05, 'epoch': 0.22}                                 
{'loss': 0.8493, 'grad_norm': 10.25, 'learning_rate': 1.5240000000000001e-05, 'epoch': 0.24}                                  
{'loss': 0.847, 'grad_norm': 13.375, 'learning_rate': 1.4840000000000002e-05, 'epoch': 0.26}                                  
{'loss': 0.8141, 'grad_norm': 15.375, 'learning_rate': 1.444e-05, 'epoch': 0.28}                                              
{'loss': 0.8033, 'grad_norm': 9.8125, 'learning_rate': 1.4040000000000001e-05, 'epoch': 0.3}                                  
{'loss': 0.8022, 'grad_norm': 12.75, 'learning_rate': 1.3640000000000002e-05, 'epoch': 0.32}                                  
{'loss': 0.7921, 'grad_norm': 11.5625, 'learning_rate': 1.3240000000000002e-05, 'epoch': 0.34}                                
{'loss': 0.7883, 'grad_norm': 11.875, 'learning_rate': 1.284e-05, 'epoch': 0.36}                                              
{'loss': 0.7699, 'grad_norm': 10.1875, 'learning_rate': 1.2440000000000001e-05, 'epoch': 0.38}                                
{'loss': 0.7697, 'grad_norm': 11.5625, 'learning_rate': 1.204e-05, 'epoch': 0.4}                                              
{'loss': 0.7682, 'grad_norm': 7.46875, 'learning_rate': 1.164e-05, 'epoch': 0.42}                                             
{'loss': 0.7831, 'grad_norm': 11.875, 'learning_rate': 1.1240000000000002e-05, 'epoch': 0.44}                                 
{'loss': 0.7728, 'grad_norm': 8.5, 'learning_rate': 1.0840000000000001e-05, 'epoch': 0.46}                                    
{'loss': 0.755, 'grad_norm': 9.375, 'learning_rate': 1.0440000000000002e-05, 'epoch': 0.48}                                   
{'loss': 0.7577, 'grad_norm': 12.4375, 'learning_rate': 1.004e-05, 'epoch': 0.5}                                              
{'loss': 0.7595, 'grad_norm': 9.25, 'learning_rate': 9.640000000000001e-06, 'epoch': 0.52}                                    
{'loss': 0.7628, 'grad_norm': 8.75, 'learning_rate': 9.240000000000001e-06, 'epoch': 0.54}                                    
{'loss': 0.7531, 'grad_norm': 7.28125, 'learning_rate': 8.84e-06, 'epoch': 0.56}                                              
{'loss': 0.7351, 'grad_norm': 10.625, 'learning_rate': 8.44e-06, 'epoch': 0.58}                                               
{'loss': 0.7546, 'grad_norm': 9.0625, 'learning_rate': 8.040000000000001e-06, 'epoch': 0.6}                                   
{'loss': 0.7306, 'grad_norm': 8.75, 'learning_rate': 7.640000000000001e-06, 'epoch': 0.62}                                    
 64%|███████████████████████████████████████████████████▌                             | 318/500 [21:52<22:16,  7.34s/it]{'loss': 0.7574, 'grad_norm': 14.125, 'learning_rate': 7.24e-06, 'epoch': 0.64}                                         
{'loss': 0.7262, 'grad_norm': 8.8125, 'learning_rate': 6.8400000000000014e-06, 'epoch': 0.66}                           
 67%|████████████████████████████████████████████▌                      | 333/500 [23:10<10:57,  3.94s/it]{'loss': 0.7345, 'grad_norm': 9.375, 'learning_rate': 6.440000000000001e-06, 'epoch': 0.68}               
{'loss': 0.7308, 'grad_norm': 8.875, 'learning_rate': 6.040000000000001e-06, 'epoch': 0.7}                
{'loss': 0.7434, 'grad_norm': 10.625, 'learning_rate': 5.64e-06, 'epoch': 0.72}                           
{'loss': 0.7303, 'grad_norm': 9.9375, 'learning_rate': 5.240000000000001e-06, 'epoch': 0.74}              
{'loss': 0.7203, 'grad_norm': 9.9375, 'learning_rate': 4.84e-06, 'epoch': 0.76}                           
{'loss': 0.7427, 'grad_norm': 13.5625, 'learning_rate': 4.440000000000001e-06, 'epoch': 0.78}             
{'loss': 0.732, 'grad_norm': 8.1875, 'learning_rate': 4.04e-06, 'epoch': 0.8}                             
{'loss': 0.7353, 'grad_norm': 7.65625, 'learning_rate': 3.6400000000000003e-06, 'epoch': 0.82}            
{'loss': 0.7283, 'grad_norm': 7.625, 'learning_rate': 3.2400000000000003e-06, 'epoch': 0.84}              
{'loss': 0.7356, 'grad_norm': 10.1875, 'learning_rate': 2.84e-06, 'epoch': 0.86}                          
{'loss': 0.7401, 'grad_norm': 10.0625, 'learning_rate': 2.4400000000000004e-06, 'epoch': 0.88}            
{'loss': 0.7151, 'grad_norm': 8.5, 'learning_rate': 2.04e-06, 'epoch': 0.9}                               
{'loss': 0.7364, 'grad_norm': 8.3125, 'learning_rate': 1.6400000000000002e-06, 'epoch': 0.92}             
{'loss': 0.7014, 'grad_norm': 9.375, 'learning_rate': 1.2400000000000002e-06, 'epoch': 0.94}              
{'loss': 0.7177, 'grad_norm': 9.5, 'learning_rate': 8.400000000000001e-07, 'epoch': 0.96}                 
{'loss': 0.7212, 'grad_norm': 6.09375, 'learning_rate': 4.4e-07, 'epoch': 0.98}                           
{'loss': 0.7286, 'grad_norm': 9.9375, 'learning_rate': 4e-08, 'epoch': 1.0}                               
{'train_runtime': 1942.2069, 'train_samples_per_second': 4.119, 'train_steps_per_second': 0.257, 'train_loss': 0.99023903465271, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████| 500/500 [32:22<00:00,  3.88s/it]

5. Saving final model...

✅ Training completed! Model saved to /project/khanhnt/control_theory/llm-jepa/compare_three_runs/stp

🎉 Fine-tuning finished successfully!
[rank0]:[W412 17:17:44.617540403 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

>>> rendezvous 127.0.0.1:29501 (set MASTER_ADDR/MASTER_PORT to override)
>>> torchrun --nproc_per_node=2 --master-addr=127.0.0.1 --master-port=29501 /project/khanhnt/control_theory/llm-jepa/stp.py --train_file /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl --output_dir /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics --num_epochs 1 --finetune_seed 42 --model_name meta-llama/Llama-3.2-1B-Instruct --learning_rate 2e-05 --batch_size 2 --grad_accum 4 --max_length 512 --last_token -2 --lbd 0.02 --predictors 0 --dynamics_tube --tube_gamma 0.95 --tube_tau 1e-3 --lbd_ts 0.02
W0412 17:17:47.389000 318806 torch/distributed/run.py:766] 
W0412 17:17:47.389000 318806 torch/distributed/run.py:766] *****************************************
W0412 17:17:47.389000 318806 torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0412 17:17:47.389000 318806 torch/distributed/run.py:766] *****************************************
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=0

1. Loading model and tokenizer...
=== Fine-tuning Script ===
Train file: /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
No eval file provided - training without evaluation
Model: meta-llama/Llama-3.2-1B-Instruct
Output: /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics
Using LoRA: False
LoRA rank: 16
Running with torchrun: world_size=2, local_rank=1
Added 11 new special tokens
Added <|mask|> token: 128267
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`

2. Loading and preparing dataset...
Loading training data from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
Loaded 8000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_train.jsonl
Map: 100%|████████████████████████████████████████████████████| 8000/8000 [00:39<00:00, 204.09 examples/s]
No evaluation file provided
Train samples: 8000
No evaluation dataset

3. Initializing representation trainer...
/project/khanhnt/control_theory/llm-jepa/stp.py:695: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `RepresentationTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Map: 100%|████████████████████████████████████████████████████| 8000/8000 [00:39<00:00, 200.63 examples/s]
/project/khanhnt/control_theory/llm-jepa/stp.py:695: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `RepresentationTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.

4. Starting training...
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
Trainer.tokenizer is now deprecated. You should use Trainer.processing_class instead.
{'loss': 39.2237, 'grad_norm': 364.0, 'learning_rate': 1.9640000000000002e-05, 'epoch': 0.02}             
{'loss': 9.2479, 'grad_norm': 420.0, 'learning_rate': 1.9240000000000002e-05, 'epoch': 0.04}              
{'loss': 7.5167, 'grad_norm': 334.0, 'learning_rate': 1.884e-05, 'epoch': 0.06}                           
{'loss': 6.5046, 'grad_norm': 316.0, 'learning_rate': 1.8440000000000003e-05, 'epoch': 0.08}              
{'loss': 5.8855, 'grad_norm': 334.0, 'learning_rate': 1.8040000000000003e-05, 'epoch': 0.1}               
{'loss': 5.5906, 'grad_norm': 372.0, 'learning_rate': 1.764e-05, 'epoch': 0.12}                           
{'loss': 5.2638, 'grad_norm': 444.0, 'learning_rate': 1.724e-05, 'epoch': 0.14}                           
{'loss': 5.1456, 'grad_norm': 504.0, 'learning_rate': 1.684e-05, 'epoch': 0.16}                           
{'loss': 5.0851, 'grad_norm': 628.0, 'learning_rate': 1.6440000000000002e-05, 'epoch': 0.18}              
{'loss': 4.8665, 'grad_norm': 684.0, 'learning_rate': 1.6040000000000002e-05, 'epoch': 0.2}               
{'loss': 4.8745, 'grad_norm': 776.0, 'learning_rate': 1.5640000000000003e-05, 'epoch': 0.22}              
{'loss': 4.8123, 'grad_norm': 844.0, 'learning_rate': 1.5240000000000001e-05, 'epoch': 0.24}              
{'loss': 4.7569, 'grad_norm': 1008.0, 'learning_rate': 1.4840000000000002e-05, 'epoch': 0.26}             
{'loss': 4.7083, 'grad_norm': 900.0, 'learning_rate': 1.444e-05, 'epoch': 0.28}                           
{'loss': 4.789, 'grad_norm': 964.0, 'learning_rate': 1.4040000000000001e-05, 'epoch': 0.3}                
{'loss': 4.7665, 'grad_norm': 1088.0, 'learning_rate': 1.3640000000000002e-05, 'epoch': 0.32}             
{'loss': 4.797, 'grad_norm': 2320.0, 'learning_rate': 1.3240000000000002e-05, 'epoch': 0.34}              
{'loss': 4.7338, 'grad_norm': 580.0, 'learning_rate': 1.284e-05, 'epoch': 0.36}                           
{'loss': 4.7779, 'grad_norm': 984.0, 'learning_rate': 1.2440000000000001e-05, 'epoch': 0.38}              
{'loss': 4.7891, 'grad_norm': 1224.0, 'learning_rate': 1.204e-05, 'epoch': 0.4}                           
{'loss': 4.8968, 'grad_norm': 3280.0, 'learning_rate': 1.164e-05, 'epoch': 0.42}                          
{'loss': 5.1986, 'grad_norm': 1888.0, 'learning_rate': 1.1240000000000002e-05, 'epoch': 0.44}             
{'loss': 5.1368, 'grad_norm': 1608.0, 'learning_rate': 1.0840000000000001e-05, 'epoch': 0.46}             
{'loss': 5.262, 'grad_norm': 3984.0, 'learning_rate': 1.0440000000000002e-05, 'epoch': 0.48}              
{'loss': 8.3592, 'grad_norm': 10624.0, 'learning_rate': 1.004e-05, 'epoch': 0.5}                          
{'loss': 6.6596, 'grad_norm': 6432.0, 'learning_rate': 9.640000000000001e-06, 'epoch': 0.52}              
{'loss': 5.7282, 'grad_norm': 8448.0, 'learning_rate': 9.240000000000001e-06, 'epoch': 0.54}              
{'loss': 5.7019, 'grad_norm': 8640.0, 'learning_rate': 8.84e-06, 'epoch': 0.56}                           
{'loss': 5.6808, 'grad_norm': 7360.0, 'learning_rate': 8.44e-06, 'epoch': 0.58}                           
{'loss': 5.6578, 'grad_norm': 3088.0, 'learning_rate': 8.040000000000001e-06, 'epoch': 0.6}               
{'loss': 5.4238, 'grad_norm': 3248.0, 'learning_rate': 7.640000000000001e-06, 'epoch': 0.62}              
{'loss': 5.5374, 'grad_norm': 4160.0, 'learning_rate': 7.24e-06, 'epoch': 0.64}                           
{'loss': 5.5175, 'grad_norm': 3920.0, 'learning_rate': 6.8400000000000014e-06, 'epoch': 0.66}             
{'loss': 5.4593, 'grad_norm': 4576.0, 'learning_rate': 6.440000000000001e-06, 'epoch': 0.68}              
{'loss': 5.4451, 'grad_norm': 7776.0, 'learning_rate': 6.040000000000001e-06, 'epoch': 0.7}               
{'loss': 5.4645, 'grad_norm': 5888.0, 'learning_rate': 5.64e-06, 'epoch': 0.72}                           
{'loss': 5.5008, 'grad_norm': 4096.0, 'learning_rate': 5.240000000000001e-06, 'epoch': 0.74}              
{'loss': 5.3543, 'grad_norm': 5120.0, 'learning_rate': 4.84e-06, 'epoch': 0.76}                           
{'loss': 5.3012, 'grad_norm': 7360.0, 'learning_rate': 4.440000000000001e-06, 'epoch': 0.78}              
{'loss': 5.2541, 'grad_norm': 4864.0, 'learning_rate': 4.04e-06, 'epoch': 0.8}                            
{'loss': 5.2753, 'grad_norm': 4192.0, 'learning_rate': 3.6400000000000003e-06, 'epoch': 0.82}             
{'loss': 5.2525, 'grad_norm': 6400.0, 'learning_rate': 3.2400000000000003e-06, 'epoch': 0.84}             
{'loss': 5.2708, 'grad_norm': 4544.0, 'learning_rate': 2.84e-06, 'epoch': 0.86}                           
{'loss': 5.2423, 'grad_norm': 4288.0, 'learning_rate': 2.4400000000000004e-06, 'epoch': 0.88}             
{'loss': 5.236, 'grad_norm': 4128.0, 'learning_rate': 2.04e-06, 'epoch': 0.9}                             
{'loss': 5.2357, 'grad_norm': 8448.0, 'learning_rate': 1.6400000000000002e-06, 'epoch': 0.92}             
{'loss': 5.1649, 'grad_norm': 4224.0, 'learning_rate': 1.2400000000000002e-06, 'epoch': 0.94}             
{'loss': 5.2203, 'grad_norm': 4416.0, 'learning_rate': 8.400000000000001e-07, 'epoch': 0.96}              
{'loss': 5.1965, 'grad_norm': 4672.0, 'learning_rate': 4.4e-07, 'epoch': 0.98}                            
{'loss': 5.2018, 'grad_norm': 4416.0, 'learning_rate': 4e-08, 'epoch': 1.0}                               
{'train_runtime': 980.6099, 'train_samples_per_second': 8.158, 'train_steps_per_second': 0.51, 'train_loss': 6.139421035766602, 'epoch': 1.0}
100%|███████████████████████████████████████████████████████████████████| 500/500 [16:20<00:00,  1.96s/it]

5. Saving final model...

✅ Training completed! Model saved to /project/khanhnt/control_theory/llm-jepa/compare_three_runs/dynamics

🎉 Fine-tuning finished successfully!
[rank0]:[W412 17:35:17.125432687 ProcessGroupNCCL.cpp:1479] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
Generating train split: 2000 examples [00:00, 207824.00 examples/s]
Loaded 2000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_test.jsonl
Map: 100%|█████████████████████████████████████████████████████| 2000/2000 [00:21<00:00, 94.92 examples/s]
PPL[regular]: 100%|█████████████████████████████████████████████████████| 500/500 [01:05<00:00,  7.63it/s]
Loaded 2000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_test.jsonl
Map: 100%|████████████████████████████████████████████████████| 2000/2000 [00:09<00:00, 203.26 examples/s]
PPL[stp]: 100%|█████████████████████████████████████████████████████████| 500/500 [01:03<00:00,  7.83it/s]
Loaded 2000 examples from /project/khanhnt/control_theory/llm-jepa/datasets/synth_test.jsonl
Map: 100%|████████████████████████████████████████████████████| 2000/2000 [00:09<00:00, 208.30 examples/s]
PPL[dynamics]: 100%|████████████████████████████████████████████████████| 500/500 [01:04<00:00,  7.72it/s]

========== Perplexity comparison ==========
  regular       PPL = 1.197361885289529
  stp           PPL = 1.1974412700548294
  dynamics      PPL = 2.9306239089258117
