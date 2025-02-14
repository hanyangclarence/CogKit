from cogmodels.finetune import register

from ..cogvideox_t2v.sft_trainer import CogVideoXT2VSftTrainer


class CogVideoX1_5T2VSftTrainer(CogVideoXT2VSftTrainer):
    pass


register("cogvideox1.5-t2v", "sft", CogVideoX1_5T2VSftTrainer)
