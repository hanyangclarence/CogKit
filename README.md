# CogKit

## run training on on Vela:
```
# 在ibm跳板机上
cd ~/mnt/embodied_o1
bash openshift/run_job2.sh openshift/train_cogkit_2nodes.yaml
```

## Eval on Vela:
```
python eval_v3.py --ckpt training_logs/v4/checkpoint-7100/ --config config/v4.yaml --dtype bfloat16  --model_name THUDM/CogVideoX1.5-5B-I2V --data_dir /gpfs/yanghan/data/rlbench_videos/test/
```