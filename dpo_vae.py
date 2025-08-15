import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, z):
        return self.activation(self.fc(z))

class ResponseDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.fc(x))
    

class VAEEncoder_2(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim  # 默认隐藏层维度等于输入
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, seq_len, hidden_size]
        h = self.activation(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class VAEDecoder_2(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, z):
        h = self.activation(self.fc1(z))
        return self.activation(self.fc2(h))  # [B, seq_len, hidden_size]

    

class GPT2(nn.Module):
    def __init__(self, latent_dim=32, gpt2_model_name="gpt2", vae_checkpoint_path=None):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        gpt2_hidden_size = self.gpt2.config.hidden_size
        gpt2_vocab_size = self.gpt2.config.vocab_size

        # Freeze GPT2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # Independent projector (same shape as GPT2 lm_head)
        self.projector = self.gpt2.lm_head

    def forward(self, input_ids, attention_mask=None):
        # 确保所有模块在 input_ids 的同一 device
        device = input_ids.device
        self.to(device)

        # GPT2 transformer
        gpt2_outputs = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = gpt2_outputs.last_hidden_state[:, -1, :]  # last token

        # Project to vocab
        logits = self.projector(hidden_state)

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=gpt2_outputs.hidden_states,
            attentions=gpt2_outputs.attentions
        )
    

class MyLatentDPOModel(nn.Module):
    def __init__(self, latent_dim=32, gpt2_model_name="gpt2", vae_checkpoint_path=None):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        gpt2_hidden_size = self.gpt2.config.hidden_size
        gpt2_vocab_size = self.gpt2.config.vocab_size

        # Freeze GPT2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # VAE components
        self.vae_encoder = VAEEncoder(gpt2_hidden_size, latent_dim)
        self.vae_decoder = VAEDecoder(latent_dim, gpt2_hidden_size)

        # Response decoder
        self.response_decoder = ResponseDecoder(gpt2_hidden_size, gpt2_hidden_size)

        # Independent projector (same shape as GPT2 lm_head)
        self.projector = self.gpt2.lm_head
        

        # ====== 额外：加载 warm-up 结果 ======
        if vae_checkpoint_path is not None:
            checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
            self.vae_encoder.load_state_dict(checkpoint["vae_encoder"])
            self.vae_decoder.load_state_dict(checkpoint["vae_decoder"])
            self.response_decoder.load_state_dict(checkpoint["response_decoder"])
            self.projector.load_state_dict(checkpoint["projector"])
            print(f"[INFO] Warm-up VAE weights loaded from {vae_checkpoint_path}")

    def forward(self, input_ids, attention_mask=None):
        # 确保所有模块在 input_ids 的同一 device
        device = input_ids.device
        self.to(device)

        # GPT2 transformer
        gpt2_outputs = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = gpt2_outputs.last_hidden_state[:, -1, :]  # last token

        # VAE path
        z, mu, logvar = self.vae_encoder(hidden_state)
        x_recon = self.vae_decoder(z)

        # Response decoder
        dec_out = self.response_decoder(x_recon)

        # Project to vocab
        logits = self.projector(dec_out)

        # hidden_states = gpt2_outputs.last_hidden_state
        # hidden_states[:,-1,:] = dec_out

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=hidden_state,
            attentions=gpt2_outputs.attentions
        )

class MyLatentDPOModel_no_decoder(nn.Module):
    def __init__(self, latent_dim=32, gpt2_model_name="gpt2", vae_checkpoint_path=None):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        gpt2_hidden_size = self.gpt2.config.hidden_size
        gpt2_vocab_size = self.gpt2.config.vocab_size

        # Freeze GPT2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # VAE components
        self.vae_encoder = VAEEncoder_2(gpt2_hidden_size, latent_dim)
        self.vae_decoder = VAEDecoder_2(latent_dim, gpt2_hidden_size)

        self.layernorm = nn.LayerNorm(gpt2_hidden_size)

        # Independent projector (same shape as GPT2 lm_head)
        # self.projector = nn.Linear(gpt2_hidden_size, gpt2_vocab_size)
        self.projector = nn.Linear(gpt2_hidden_size, gpt2_vocab_size)
        with torch.no_grad():
            self.projector.weight.copy_(self.gpt2.lm_head.weight)
            if self.projector.bias is not None:
                self.projector.bias.zero_()  # 可选：初始化偏置为 0
        # 确保可训练
        self.projector.weight.requires_grad = True
        self.projector.bias.requires_grad = True

        # ====== 额外：加载 warm-up 结果 ======
        if vae_checkpoint_path is not None:
            checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
            self.vae_encoder.load_state_dict(checkpoint["vae_encoder"])
            self.vae_decoder.load_state_dict(checkpoint["vae_decoder"])
            self.projector.load_state_dict(checkpoint["projector"])
            print(f"[INFO] Warm-up VAE weights loaded from {vae_checkpoint_path}")

    def forward(self, input_ids, attention_mask=None):
        # 确保所有模块在 input_ids 的同一 device
        device = input_ids.device
        self.to(device)

        # GPT2 transformer
        gpt2_outputs = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = gpt2_outputs.last_hidden_state[:, -1, :]  # last token

        # VAE path
        z, mu, logvar = self.vae_encoder(hidden_state)
        x_recon = self.vae_decoder(z)

        # x_recon = hidden_state + 0.5 * x_recon

        # test
        x_recon = hidden_state

        x_recon = self.layernorm(x_recon)

        # Project to vocab
        logits = self.projector(x_recon)

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=gpt2_outputs.hidden_states,
            attentions=gpt2_outputs.attentions
        )
    

# add layernorm, no response decoder, all tokens in VAE
class GPT2VAEModel(nn.Module):
    def __init__(self, latent_dim=32, gpt2_model_name="gpt2"):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        gpt2_hidden_size = self.gpt2.config.hidden_size

        # Freeze GPT2 parameters
        for param in self.gpt2.parameters():
            param.requires_grad = False

        # VAE
        self.vae_encoder = VAEEncoder_2(gpt2_hidden_size, latent_dim)
        self.vae_decoder = VAEDecoder_2(latent_dim, gpt2_hidden_size)

        # LayerNorm before lm_head
        self.layernorm = nn.LayerNorm(gpt2_hidden_size)

        gpt2_vocab_size = self.gpt2.config.vocab_size
        # lm_head
        self.projector = nn.Linear(gpt2_hidden_size, gpt2_vocab_size)
        with torch.no_grad():
            self.projector.weight.copy_(self.gpt2.lm_head.weight)
            if self.projector.bias is not None:
                self.projector.bias.zero_()  # 可选：初始化偏置为 0
        # 确保可训练
        self.projector.weight.requires_grad = True
        self.projector.bias.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        device = input_ids.device
        self.to(device)

        # 1. GPT2 transformer
        gpt2_outputs = self.gpt2.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = gpt2_outputs.last_hidden_state  # [B, seq_len, hidden_size]

        # 2. VAE
        z, mu, logvar = self.vae_encoder(hidden_states)
        hidden_recon = self.vae_decoder(z)  # [B, seq_len, hidden_size]

        # 3. LayerNorm
        hidden_recon = self.layernorm(hidden_recon)

        # 4. Project to vocab
        logits = self.projector(hidden_recon)  # [B, seq_len, vocab_size]

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            hidden_states=gpt2_outputs.hidden_states,
            attentions=gpt2_outputs.attentions
        )