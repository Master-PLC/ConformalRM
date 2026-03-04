import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import datasets
import torch

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
import numpy as np
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def process_item(indexed_item, data_name):
    index, item = indexed_item
    if data_name == 'hs':
        return [{
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ],
            "rating": [item['helpfulness']]
        }], index

    elif data_name == 'saferlhf':
        return [{
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response_0"]}
            ],
            "rating": [item["response_0_severity_level"]]
        }, {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response_1"]}
            ],
            "rating": [item["response_1_severity_level"]]
        }], index

    elif data_name == 'ufb':
        return [{
            "messages": item["chosen"],
            "rating": [item["score_chosen"]]
        }, {
            "messages": item["rejected"],
            "rating": [item["score_rejected"]]
        }], index

    elif data_name == 'hhrlhf':
        remove_tag = lambda x: x.replace("Human: ", "").replace("Assistant: ", "")
        chosen = item["chosen"].strip()
        chosen_messages = chosen.split("\n\n")
        chosen_messages = [remove_tag(msg) for msg in chosen_messages]
        chosen_messages = [
            {"role": "user", "content": msg} if i % 2 == 0 else {"role": "assistant", "content": msg}
            for i, msg in enumerate(chosen_messages)
        ]

        rejected = item["rejected"].strip()
        rejected_messages = rejected.split("\n\n")
        rejected_messages = [remove_tag(msg) for msg in rejected_messages]
        rejected_messages = [
            {"role": "user", "content": msg} if i % 2 == 0 else {"role": "assistant", "content": msg}
            for i, msg in enumerate(rejected_messages)
        ]

        return [{
            "messages": chosen_messages,
            "rating": [1.0]
        }, {
            "messages": rejected_messages,
            "rating": [0.0]
        }], index

    elif data_name == 'shp':
        score_ratio = item['score_ratio']
        if score_ratio <= 2:
            return [], index

        return [{
            "messages": [
                {"role": "user", "content": item["history"]},
                {"role": "assistant", "content": item["human_ref_A"]}
            ],
            "rating": [1.0]
        }, {
            "messages": [
                {"role": "user", "content": item["history"]},
                {"role": "assistant", "content": item["human_ref_B"]}
            ],
            "rating": [0.0]
        }], index

    elif data_name == 'armorm':
        attributes = [
            "helpsteer-helpfulness", "helpsteer-correctness", "helpsteer-coherence",
            "helpsteer-complexity", "helpsteer-verbosity", "ultrafeedback-overall_score",
            "ultrafeedback-instruction_following", "ultrafeedback-truthfulness",
            "ultrafeedback-honesty", "ultrafeedback-helpfulness", "beavertails-is_safe",
            "prometheus-score", "argilla-overall_quality", "argilla-judge_lm",
            "code-complexity", "code-style", "code-explanation",
            "code-instruction-following", "code-readability",
        ]
        return [{
            "messages": item["messages"],
            "rating": [item.get(attr) for attr in attributes]
        }], index


def process_dataset(data_name, subset_name, base_path="./rawdata", num_workers=30):
    dataset_path = os.path.join(base_path, data_name)
    data = datasets.load_from_disk(dataset_path)[subset_name]
    indexed_data = list(enumerate(data))

    worker_func = partial(process_item, data_name=data_name)
    with Pool(processes=num_workers) as pool:
        processed_list = list(tqdm(pool.imap(worker_func, indexed_data), total=len(indexed_data), desc="Processing raw data"))
    processed_list = [x for x in processed_list if x]
    processed_list = sorted(processed_list, key=lambda x: x[1])
    processed_list = [x[0] for x in processed_list]
    processed_list = sum(processed_list, [])

    print(f"Finished processing. Total items created: {len(processed_list)}")
    return processed_list


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/tidalfs-bdsz01/dataset/llm_dataset/plc_data/ckpts/FsfairX-LLaMA3-RM-v0.1")
    parser.add_argument("--rawdata_dir", type=str, default="./rawdata")
    parser.add_argument("--output_dir", type=str, default="./embeddings/normal")
    parser.add_argument("--data_name", type=str, default="hs")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1)

    args = parser.parse_args()

    rm = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    ds = process_dataset(args.data_name, args.subset, base_path=args.rawdata_dir, num_workers=args.num_workers)
    ds = ds[:args.max_samples] if args.max_samples > 0 else ds

    model_name = args.model_path.split("/")[-1]
    if model_name.startswith("LLaMA"):
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm.config.pad_token_id = rm_tokenizer.pad_token_id

    def collate_fn(batch):
        # 这个函数负责将单个样本聚合成一个批次
        formatted_texts = []
        batch_labels = []

        for example in batch:
            # 格式化文本
            if model_name.endswith("FsfairX-LLaMA3-RM-v0.1"):
                conv_formatted = rm_tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                ).replace(rm_tokenizer.bos_token, "")
            else:
                conv_formatted = rm_tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )
            formatted_texts.append(conv_formatted)

            # 提取标签
            label = example["rating"]
            label = [np.nan if l is None else l for l in label]
            batch_labels.append(label)

        # 对整个批次的文本进行tokenize，并进行padding
        tokenized_batch = rm_tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True, # 关键：开启padding
            truncation=True, # 关键：开启截断，防止超长
            max_length=4096 # 根据模型和显存调整
        )
        return tokenized_batch, torch.tensor(batch_labels, dtype=torch.float32)

    dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)

    embeddings = []
    labels = []
    for batch_tokenized, batch_labels in tqdm(dataloader, desc="Processing dataset in batches"):
        # 将数据移动到主GPU，DataParallel会自动分发
        batch_tokenized = {k: v.to('cuda') for k, v in batch_tokenized.items()}

        with torch.no_grad():
            outputs = rm(**batch_tokenized)

            # 提取每个样本的最后一个token的embedding
            # `attention_mask`帮助我们找到每个序列真正的结尾
            last_token_indices = batch_tokenized['attention_mask'].sum(dim=1) - 1
            last_hidden_states = outputs.last_hidden_state

            # 使用高级索引一次性提取所有样本的最后一个token的embedding
            batch_embeddings = last_hidden_states[torch.arange(last_hidden_states.shape[0]), last_token_indices]

            embeddings.append(batch_embeddings.cpu())

        labels.append(batch_labels)

    # 合并所有批次的结果
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    labels = labels.squeeze(1)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = f"{args.output_dir}/{model_name}_{args.data_name}_{args.subset}.safetensors"
    save_file({"embeddings": embeddings, "labels": labels}, save_path)
    print(f"Saved embeddings to {save_path}")
