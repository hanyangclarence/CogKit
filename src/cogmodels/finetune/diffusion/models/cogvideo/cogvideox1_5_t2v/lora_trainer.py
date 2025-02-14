from cogmodels.finetune import register

from ..cogvideox_t2v.lora_trainer import CogVideoXT2VLoraTrainer


class CogVideoX1_5T2VLoraTrainer(CogVideoXT2VLoraTrainer):
    pass


register("cogvideox1.5-t2v", "lora", CogVideoX1_5T2VLoraTrainer)
