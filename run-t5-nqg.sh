python3 train-t5-nqg.py \
  --model_name_or_path t5-small \
  --config_name t5-small \
  --output_dir checkpoints/t5.nqg \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --max_steps 10000 \
  --save_steps 2500 \
  --eval_steps 2500 \
  --do_train
