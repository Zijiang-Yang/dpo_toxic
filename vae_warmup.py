import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from dpo_vae import MyLatentDPOModel_no_decoder, MyLatentDPOModel # 你的 VAE 模型


model_name = "gpt2-medium"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. 加载数据集并筛选单轮对话
# ---------------------------
dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")

print(dataset[1]["messages"])
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
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
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
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# batch tokenization
batch_size = 8
all_input_ids = []
all_attention_mask = []
all_labels = []

for i in range(0, len(single_turn_examples), batch_size):
    batch = single_turn_examples[i:i+batch_size]
    tokenized = tokenize_function(batch)
    all_input_ids.append(tokenized["input_ids"])
    all_attention_mask.append(tokenized["attention_mask"])
    all_labels.append(tokenized["labels"])

dataset_pt = SingleTurnDataset({
    "input_ids": torch.cat(all_input_ids),
    "attention_mask": torch.cat(all_attention_mask),
    "labels": torch.cat(all_labels)
})

dataloader = DataLoader(dataset_pt, batch_size=batch_size, shuffle=True)

# ---------------------------
# 4. 初始化模型
# ---------------------------
model = MyLatentDPOModel_no_decoder(gpt2_model_name=model_name).to(device)
# model = MyLatentDPOModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 5. Warm-up 训练循环
# ---------------------------
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        labels = labels[:, 0]  # 只取第一个 token
        

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]

        # 使用 CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        # logits: (batch, seq_len, vocab), labels: (batch, seq_len)
        # loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss = loss_fct(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

print("Warm-up finished!")

# torch.save({
#     'vae_encoder': model.vae_encoder.state_dict(),
#     'vae_decoder': model.vae_decoder.state_dict(),
#     'response_decoder': model.response_decoder.state_dict(),
#     'projector': model.projector.state_dict()
# }, "vae_warmup.pth")

torch.save({
    'vae_encoder': model.vae_encoder.state_dict(),
    'vae_decoder': model.vae_decoder.state_dict(),
    'projector': model.projector.state_dict()
}, "vae_warmup_no_decoder_medium.pth")

