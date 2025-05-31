# CogKit

## Train on Vela:
```
bash run_i2v_zero_vela.sh --config config/v4.yaml --output_dir training_logs/v4
```

## Eval on Vela:
```
python eval_v3.py --ckpt training_logs/v4/checkpoint-7100/ --config config/v4.yaml --dtype bfloat16  --model_name THUDM/CogVideoX1.5-5B-I2V --data_dir /gpfs/yanghan/data/rlbench_videos/test/
```