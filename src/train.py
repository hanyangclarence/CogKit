import argparse
import pdb
from omegaconf import OmegaConf
from cogkit.utils.utils import get_obj_from_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--training_type", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args, unknown = parser.parse_known_args()

    if args.debug:
        pdb.set_trace()
    
    config = OmegaConf.load(args.config)

    trainer_cls = get_obj_from_str(config["trainer_class"])
    trainer = trainer_cls()
    trainer.fit()


if __name__ == "__main__":
    main()
