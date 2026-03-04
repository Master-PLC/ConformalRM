import os
ROOT = os.path.dirname(os.path.abspath(__file__))

from argparse import ArgumentParser
from datasets import load_dataset

RAW_DATA_DIR = f'{ROOT}/rawdata'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_name", type=str, default="hs")
    args = parser.parse_args()

    local_save_path = f'{ROOT}/rawdata/{args.data_name}'
    loading_map = {
        'hs': lambda: load_dataset("nvidia/HelpSteer"),
        'ufb': lambda: load_dataset("HuggingFaceH4/ultrafeedback_binarized"),
        'saferlhf': lambda: load_dataset("PKU-Alignment/PKU-SafeRLHF", "default"),
    }
    dataset = loading_map[args.data_name]()

    print(f"Saving the dataset to '{local_save_path}'...")
    dataset.save_to_disk(local_save_path)
    print("Dataset saved successfully!")
