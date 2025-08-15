import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from dpo_vae import MyLatentDPOModel_no_decoder, MyLatentDPOModel, GPT2VAEModel  # 你的 VAE 模型

model_name = "gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. 加载数据集并筛选单轮对话
# ---------------------------
dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")

single_turn_examples = []
for i in range(len(dataset)):
    messages = dataset[i]["messages"]
    if len(messages) >= 2:
        single_turn_examples.append({
            "prompt": messages[0]["content"],
            "response": messages[1]["content"]
        })

print(f"Number of single-turn examples: {len(single_turn_examples)}")

# ---------------------------
# 2. 初始化 tokenizer
# ---------------------------
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 避免padding问题

# 加上特殊标记（可选）
BOS = "<BOS>"
SEP = "<SEP>"
tokenizer.add_tokens([BOS, SEP])
bos_id = tokenizer.convert_tokens_to_ids(BOS)
sep_id = tokenizer.convert_tokens_to_ids(SEP)

# ---------------------------
# 3. Tokenize 成自回归格式
# ---------------------------
def tokenize_function(examples, max_length=128):
    texts = []
    for ex in examples:
        # 拼接 prompt 和 response
        text = f"{BOS} {ex['prompt']} {SEP} {ex['response']}"
        texts.append(text)

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()

    # mask 掉 prompt 部分 loss
    for i, ex in enumerate(examples):
        # 找到 SEP 的位置
        sep_positions = (input_ids[i] == sep_id).nonzero(as_tuple=True)
        if len(sep_positions[0]) > 0:
            sep_pos = sep_positions[0].item()
            labels[i, :sep_pos+1] = -100  # prompt + SEP 不算 loss

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# PyTorch Dataset
class SingleTurnDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# 创建 DataLoader
batch_size = 8
tokenized_batches = []
for i in range(0, len(single_turn_examples), batch_size):
    batch = single_turn_examples[i:i+batch_size]
    tokenized = tokenize_function(batch)
    tokenized_batches.append(tokenized)

dataset_pt = SingleTurnDataset([
    {k: v[idx] for k, v in tokenized.items()}
    for tokenized in tokenized_batches
    for idx in range(tokenized["input_ids"].size(0))
])

dataloader = DataLoader(dataset_pt, batch_size=batch_size, shuffle=True)

print("Data loading finished")

# ---------------------------
# 4. 初始化模型
# ---------------------------
model = GPT2VAEModel(gpt2_model_name=model_name).to(device)
model.gpt2.resize_token_embeddings(len(tokenizer))  # 加了新 token 需要调整词表

optimizer = torch.optim.Adam(model.gpt2.parameters(), lr=1e-3)

# ---------------------------
# 5. Warm-up 训练循环（自回归）
# ---------------------------
num_epochs = 50
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  # 忽略 mask 部分

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]  # (batch, seq_len, vocab)

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

print("Warm-up finished!")

torch.save({
    'vae_encoder': model.vae_encoder.state_dict(),
    'vae_decoder': model.vae_decoder.state_dict(),
    'projector': model.projector.state_dict()
}, "gpt2vae_warmup.pth")
