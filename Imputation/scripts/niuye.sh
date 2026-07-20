for model_name in LLM4Log
do
#32 64 128 256
for batch_size in 16
do
for lr in 0.001
do
for gpt_layer in 3
do
for enc_in in 6
do




python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path niuye1-5hf.csv \
  --model_id niuye1_5hf_mask_0.2 \
  --mask_rate 0.2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 8 \
  --label_len 0 \
  --pred_len 0 \
  --patch_size 1 \
  --d_ff 768 \
  --stride 1 \
  --gpt_layer $gpt_layer \
  --d_model 768 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 6 \
  --batch_size $batch_size \
  --train_epoch 30 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr 

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path niuye1-5hf.csv \
  --model_id niuye1_5hf_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 8 \
  --label_len 0 \
  --pred_len 0 \
  --patch_size 1 \
  --d_ff 768 \
  --stride 1 \
  --gpt_layer $gpt_layer \
  --d_model 768 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 6 \
  --batch_size $batch_size \
  --train_epoch 30 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr 

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path niuye1-5hf.csv \
  --model_id niuye1_5hf_mask_0.7 \
  --mask_rate 0.7 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 8 \
  --label_len 0 \
  --pred_len 0 \
  --patch_size 1 \
  --d_ff 768 \
  --stride 1 \
  --gpt_layer $gpt_layer \
  --d_model 768 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 6 \
  --batch_size $batch_size \
  --train_epoch 30 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr 

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path niuye1-5hf.csv \
  --model_id niuye1_5hf_mask_0.9 \
  --mask_rate 0.9 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 8 \
  --label_len 0 \
  --pred_len 0 \
  --patch_size 1 \
  --d_ff 768 \
  --stride 1 \
  --gpt_layer $gpt_layer \
  --d_model 768 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out 6 \
  --batch_size $batch_size \
  --train_epoch 30 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr 
done
done
done
done
done