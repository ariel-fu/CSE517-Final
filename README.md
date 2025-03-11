
# **Human Think Language Reproduction** 
This repo contains the code and data for a replication for the paper: "[How Do Humans Write Code? Large Models Do It the Same Way Too](https://arxiv.org/pdf/2402.15729)". The original authors released their CodeLlama model at https://huggingface.co/seamoke111/HTL-CodeLlama-7B and their original code at https://github.com/seamoke/Human-Think-Language/.

## **Introduction**
Hello! This is Ariel and Alex Fu's final project for CSE 517, Winter 2025. All of our modifications have been made in the `colab` branch. For this project, we completed the building, training, and evaluation of the model in Google Colab: https://colab.research.google.com/drive/1QgkGahjRLJOcYC65j9Io3uROB4iM11aZ?usp=sharing. The notebook contains all of the code for cloning the correct branch of the repo, installing the dependencies, setting HuggingFace credentials, training the model, and evaluating the model. 

## **Pretrained Model**
The reproduced model uses the [MAmmoTH Llama Code Base 7B](https://huggingface.co/TIGER-Lab/MAmmoTH-Coder-7B) model.

## **Installation**
The required packages are listed in requirements.txt. We have a cell in our Google Colab notebook under "SET UP" that completes the setup of the Python library dependencies.

## **How to Use**
## 1. Data
The data for training and testing was provided by the authors in their original [GitHub](https://github.com/seamoke/Human-Think-Language/). We have also included them in this repo.
All of the data for training and testing is available in `math_eval/dataset`. The data for training is in the file `math_eval/dataset/train_llama_data/llama_data_final32l.jsonl`.
The authors provided several datasets. For our project, we only used the following datasets, following the order in the repo and the same styling:
- SVAMP
- asdiv
- deepmind
- gsm8k
- math
- mawps
- numglue
- simuleq

## 2. Authentication
The HTL model is based off the MAmmoTH Code Llama 7B model. The user must be authenticated into HuggingFace to properly fetch the model and train the HTL model. To evaluate the re-produced model, the user must also be authenticated into HuggingFace to properly fetch the trained model base and weights from HuggingFace.

In the Google Colab, we have a dedicated cell to run to authenticate into HuggingFace. In the event that the user does not have a token for HuggingFace, one is provided in the comments. 

## 3. Train
We run the following script in a cell on Google Colab under the section "Run the training". The script runs the training code for 1 epoch, with the hyperparameters discussed in the final report.

python -m torch.distributed.run \
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
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 8 \
 --evaluation_strategy "steps" \
 --eval_steps 0.20 \
 --save_strategy "steps" \
 --save_steps 100 \
 --learning_rate "2e-5" \
 --weight_decay 0. \
 --warmup_ratio 0.03 \
 --lr_scheduler_type "cosine" \
 --save_safetensors True  \
 --flash_attn False \
 --logging_steps 1 \
 --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer


## 4. Eval
We run the following script in a cell in Google Colab, after specifying a dataset. Note that the evaluation code directly uses the model that has been saved to HuggingFace, not the model from the previous training (unless the model reaches step 300 and gets saved onto HuggingFace). The updated evaluation code retrieves the model from HuggingFace, provided the user authenticates into HuggingFace. We also provide the links to the reproduced model uploaded to HuggingFace in the report. 

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

python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'dataset_name' \
    --output '/content/CSE517-Final/outputs/dataset_name.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print
