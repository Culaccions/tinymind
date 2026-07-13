from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 对文本进行分词转化成input_id
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 自己添加BOS和EOS
            truncation=True,  # 长度超过max_length的自动截断
        ).input_ids
        # 添加BOS和EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # 不足最大长度的填充到最大长度
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        # 转换为张量
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 编写labels，防止PAD参加loss计算
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        # 编写attn_mask，将PAD位置设为0，非PAD位置设为1
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {"input_ids": input_ids, "labels": labels, "attn_mask": attn_mask}
