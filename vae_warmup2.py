import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import math
from tqdm import tqdm

# ===== 你自己的模型 =====
from dpo_vae import GPT2VAEModel  # 假设这是你写的模型

model_name = "gpt2"
latent_size = 64
batch_size = 4
lr = 5e-5
max_length = 128


# ===== 配置 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
train_batch_size = 4
learning_rate = 5e-5
warmup_steps = 500
max_grad_norm = 1.0

# ===== 初始化 =====
model = GPT2VAEModel()
model.to(device)


# ===== 数据加载 =====
dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")

single_turn_examples = []
for i in range(len(dataset)):
    messages = dataset[i]["messages"]
    single_turn_examples.append({
        "prompt": messages[0]["content"],
        "response": messages[1]["content"]
    })

print(f"Number of single-turn examples: {len(single_turn_examples)}")



# ---------------------------
# 2. 初始化 tokenizer
# ---------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 避免padding问题

# ---------------------------
# 3. Tokenize
# ---------------------------
def tokenize_function(examples):
    encodings = tokenizer(
        [ex["prompt"] for ex in examples],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    decodings = tokenizer(
        [ex["response"] for ex in examples],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"],
            "labels": decodings["input_ids"]}

# 手动创建 PyTorch Dataset
class SingleTurnDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer= GPT2Tokenizer.from_pretrained("gpt2"), max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encodings = self.tokenizer(
            ex["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        decodings = self.tokenizer(
            ex["response"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": decodings["input_ids"].squeeze(0)
        }
    
tokenized = tokenize_function(single_turn_examples)
dataset_pt = SingleTurnDataset(tokenized)
train_loader = DataLoader(dataset_pt, batch_size=batch_size, shuffle=True)

# # batch tokenization
# batch_size = 8
# all_input_ids = []
# all_attention_mask = []
# all_labels = []

# for i in range(0, len(single_turn_examples), batch_size):
#     batch = single_turn_examples[i:i+batch_size]
#     tokenized = tokenize_function(batch)
#     all_input_ids.append(tokenized["input_ids"])
#     all_attention_mask.append(tokenized["attention_mask"])
#     all_labels.append(tokenized["labels"])

# dataset_pt = SingleTurnDataset({
#     "input_ids": torch.cat(all_input_ids),
#     "attention_mask": torch.cat(all_attention_mask),
#     "labels": torch.cat(all_labels)
# })

# train_loader = DataLoader(dataset_pt, batch_size=1, shuffle=True)


# ===== 优化器 & scheduler =====
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# ===== 训练循环 =====
model.train()
for epoch in range(epochs):
    total_loss = 0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # 前向传播
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        labels = batch["labels"].to(device)

        # 如果需要在 lm_head 之前做 LayerNorm
        # if final_ln is not None:
        #     outputs.logits = model.lm_head(final_ln(outputs.hidden_states[-1]))

        logits = outputs["logits"]

        # 使用 CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        # logits: (batch, seq_len, vocab), labels: (batch, seq_len)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # loss = loss_fct(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

print("训练完成 ✅")
torch.save({
    'vae_encoder': model.vae_encoder.state_dict(),
    'vae_decoder': model.vae_decoder.state_dict(),
    'projector': model.projector.state_dict()
}, "vae_warmup2.pth")
