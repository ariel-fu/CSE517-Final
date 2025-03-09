python math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'asdiv' \
    --output 'outputs/asdiv.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print

python math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'gsm8k' \
    --output '/content/CSE517-Final/outputs/gsm8k.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print

python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'svamp' \
    --output '/content/CSE517-Final/outputs/svamp.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print

python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'math' \
    --output '/content/CSE517-Final/outputs/math.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print

python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'numglue' \
    --output '/content/CSE517-Final/outputs/numglue.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print

python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'simuleq' \
    --output '/content/CSE517-Final/outputs/simuleq.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print

python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'deepmind' \
    --output '/content/CSE517-Final/outputs/deepmind.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print

python /content/CSE517-Final/math_eval/run_open_htl.py \
    --model "htl_model" \
    --shots 0 \
    --stem_flan_type "pot_prompt" \
    --batch_size 8 \
    --dataset 'mawps' \
    --output '/content/CSE517-Final/outputs/mawps.jsonl' \
    --model_max_length 3072 \
    --cot_backup \
    --print
