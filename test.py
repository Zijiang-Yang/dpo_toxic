import torch
from transformers import GPT2Tokenizer
import torch.nn.functional as F
from dpo_vae import GPT2,MyLatentDPOModel,MyLatentDPOModel_no_decoder  # 你的模型
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from omegaconf import OmegaConf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.5, top_k=50, top_p=0.95, device="cpu"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        # ====== 1. 完整 GPT2 forward（包括 embedding） ======
        # outputs = model.gpt2.transformer(input_ids, output_attentions=True, output_hidden_states=True)
        # hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_dim)

        # # ====== 2. Final layernorm ======
        # hidden_states = model.gpt2.transformer.ln_f(hidden_states)

        # # ====== 3. lm_head logits（作用于所有 token） ======
        # logits = model.lm_head(hidden_states)  # (B, seq_len, vocab_size)
        # logits = logits[:, -1, :] / temperature     # 只取最后一个 token

        outputs = model(input_ids)
        logits = outputs["logits"]/temperature

        # ====== 4. Top-k filtering ======
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(1)
            logits = torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)

        # ====== 5. Top-p (nucleus) filtering ======
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')

        # ====== 6. 采样下一个 token ======
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # ====== 7. 拼接到输入序列 ======
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # ====== 8. 提前停止 ======
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ===== 使用示例 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
# model = GPT2().to(device)
model = MyLatentDPOModel_no_decoder(gpt2_model_name="gpt2-medium",vae_checkpoint_path="vae_warmup_no_decoder_medium.pth")

prompt = "What is the color of an apple"
output = generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9, device=device)
print(output)


@torch.no_grad()
def generate_text2(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, device="cpu"):
    """
    自写生成循环，支持 EOS token 停止
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # last token logits
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # 如果生成了 EOS token，则提前停止
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# # # =========================
# # 使用 gpt2-median
# # =========================
# tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
# model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
# model.to("cpu")  # 或 "cuda" 如果有 GPU

# prompt = "What is the color of an apple"
# output = generate_text2(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, device="cpu")
# print(output)




# prompt = "What is the color of an apple"
# input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# # 3. 生成文本
# # 可以设置 max_new_tokens, temperature, top_k, top_p 来控制生成
# output_ids = model.generate(
#     input_ids,
#     max_new_tokens=50,
#     temperature=0.8,
#     top_k=50,
#     top_p=0.9,
#     do_sample=True,
#     eos_token_id=tokenizer.eos_token_id
# )

# # 4. 解码输出
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(output_text)
