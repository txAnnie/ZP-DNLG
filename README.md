## Description

This repo. is prepared for the conference paper "**Coupling Context Modeling with Zero Pronoun Recovering for Document-Level Natural Language Generation (EMNLP-2021)**". We built the project based on the pytorch version of OpenNMT (v0.2.1). 

## Usage

1. Processing Source Data with Zero Pronoun Position Detected

The source data of pro-drop language first processed with zero pronoun position detected. For this zero pronoun preprocess period, we provide the trained model on Chinese language in DPro_model. 

2. Training NMT Model

**Step 1: Data Preprocessing**

```
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

**Step 2: Model Training**

*Training the sentence-level NMT baseline:*

```
python train.py -data path_to_preprocessed_data/data -freeze_d True -save_model path_to_saved_model/model -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 300000 -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot -label_smoothing 0.1 -valid_steps 30000 -save_checkpoint_steps 30000 
```

*Training the document-level NMT with zero pronoun recovered:*

```
python train.py -data path_to_preprocessed_data/data -freeze_d False -train_from path_to_pretrained_sentence-level_model -save_model path_to_saved_model/model -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 300000 -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate2 -max_grad_norm 0 -param_init 0 -param_init_glorot -label_smoothing 0.1 -valid_steps 200 -save_checkpoint_steps 200
```

*Translation*

```
python translate.py -model path_to_trained_document-level_model -src path_to_source_data -output path_to_save_output 
```
