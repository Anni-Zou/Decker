#export CUDA_VISIBLE_DEVICES=0

DATA="creak"
MODEL="microsoft/deberta-v3-large"
LR="9e-6"
BS="32"
ML="128"
PD_BS="16"
SEED="666666"

OUTPUT="${MODEL}_${DATA}_lr${LR}_bs${PD_BS}_ml${ML}_seed${SEED}"

python run_confact.py \
  --task ${DATA} \
  --train_path data/${DATA}/graph/train_graph_numnode20.jsonl \
  --dev_path data/${DATA}/graph/dev_graph_numnode20.jsonl \
  --test_path data/${DATA}/graph/dev_graph_numnode20.jsonl \
  --num_train_epochs 5 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --load_best_model_at_end true \
  --model_name_or_path ${MODEL} \
  --max_seq_length ${ML} \
  --learning_rate ${LR} \
  --per_device_train_batch_size ${PD_BS} \
  --per_device_eval_batch_size 32 \
  --output_dir saved_outputs_my/${OUTPUT} \
  --do_train \
  --do_eval \
  --do_predict \
  --seed ${SEED} \
  --contra_path data/${DATA}/graph/contrast_graph_numnode20.jsonl