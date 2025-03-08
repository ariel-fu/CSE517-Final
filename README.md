
# **Human Think Language** 
This repo contains the code and data for "[How Do Humans Write Code? Large Models Do It the Same Way Too](https://arxiv.org/pdf/2402.15729)"

The original authors released their CodeLlama model at: https://huggingface.co/seamoke111/HTL-CodeLlama-7B.
## **Installation**
The required packages are listed in requirements.txt.
We have also set up a notebook in Google Colab, where we ran our full experiments.
https://colab.research.google.com/drive/1QgkGahjRLJOcYC65j9Io3uROB4iM11aZ?usp=sharing

## **How to Use**
## 1. Data
All of the data is available in `math_eval/dataset`

## 2.Train
We run the following in a cell in Google Colab. It can also be run on a terminal, if `\ \n` is replaced with `` in a notepad. It is also important to note that we are using the directory format of Google Colab, and paths are prefixed with `\content\`. The model is checkpoint-ed every 1200 steps.

!CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
 --nproc_per_node=1 \
 --master_port=6066 \
 --nnodes=1 \
 --master_addr=localhost \
 --node_rank=0 \
 /content/CSE517-Final/htl/train_htl.py \
 --model_name_or_path "TIGER-Lab/MAmmoTH-Coder-7B" \
 --data_path "/content/CSE517-Final/math_eval/dataset/train_llama_data/llama_data_final32k.jsonl" \
 --output_dir "/content/checkpoint/codellama_final32k${learning_rate}" \
 --num_train_epochs 1 \
 --per_device_train_batch_size 4 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 8 \
 --evaluation_strategy "steps" \
 --eval_steps 0.20 \
 --save_strategy "steps" \
 --save_steps 1200 \
 --learning_rate "2e-5" \
 --weight_decay 0. \
 --warmup_ratio 0.03 \
 --lr_scheduler_type "cosine" \
 --save_safetensors True  \
 --flash_attn False \
 --logging_steps 1 \
 --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer


## 3.Eval
We run the following in a cell in Google Colab. Note that the evaluation code directly uses the model that has been saved to HuggingFace. The updated evaluation code retrieves the model from HuggingFace, provided the user logs into HuggingFace. We also provide the links to the HuggingFace models in the report. 

To run the code snippet below in terminal, please update the path to `run_open_htl.py`, specifically the prefix of `CSE517-Final/math_eval/run_open_htl.py`. Please also update the dataset name placeholders (`dataset_name`) with the intended dataset. 

We support evaluation on the following datasets: 
1. GSM8K
2. NumGLUE
3. MATH
4. SimulEq
5. DeepMind
6. SVAMP
7. MAWPS
8. ASDiv

If an invalid dataset is requested, the evaluation throws an error and quits.

!python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'dataset_name' \
    --output '/content/CSE517-Final/outputs/dataset_name.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print
