huggingface-cli login
python -m torch.distributed.run \
 --nproc_per_node=1 \
 --master_port=6066 \
 --nnodes=1 \
 --master_addr=localhost \
 --node_rank=0 \
 htl/train_htl.py \
 --model_name_or_path "TIGER-Lab/MAmmoTH-Coder-7B" \
 --data_path "math_eval/dataset/train_llama_data/llama_data_final32k.jsonl" \
 --output_dir "checkpoint/codellama_final32k${learning_rate}" \
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
