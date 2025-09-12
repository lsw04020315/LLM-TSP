import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils.rev_in import RevIn
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType



class ComplexLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ComplexLinear, self).__init__()
        self.fc_real = nn.Linear(input_dim, output_dim)
        self.fc_imag = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x_real = torch.real(x)
        x_imag = torch.imag(x)
        out_real = self.fc_real(x_real) - self.fc_imag(x_imag)
        out_imag = self.fc_real(x_imag) + self.fc_imag(x_real)
        return torch.complex(out_real, out_imag)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


class MultiFourier(torch.nn.Module):
    def __init__(self, N, P):
        super(MultiFourier, self).__init__()
        self.N = N
        self.P = P
        self.a = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(max(N), len(N)), requires_grad=True)

    def forward(self, t):
        output = torch.zeros_like(t)
        t = t.unsqueeze(-1).repeat(1, 1, max(self.N))  # shape: [batch_size, seq_len, max(N)]
        n = torch.arange(max(self.N)).unsqueeze(0).unsqueeze(0).to(t.device)  # shape: [1, 1, max(N)]
        for j in range(len(self.N)):  # loop over seasonal components
            # import ipdb; ipdb.set_trace()
            cos_terms = torch.cos(2 * np.pi * (n[..., :self.N[j]] + 1) * t[..., :self.N[j]] / self.P[
                j])  # shape: [batch_size, seq_len, N[j]]
            sin_terms = torch.sin(2 * np.pi * (n[..., :self.N[j]] + 1) * t[..., :self.N[j]] / self.P[
                j])  # shape: [batch_size, seq_len, N[j]]
            output += torch.matmul(cos_terms, self.a[:self.N[j], j]) + torch.matmul(sin_terms, self.b[:self.N[j], j])
        return output


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class TSP(nn.Module):

    def __init__(self, configs, device):
        super(TSP, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.pred_len = configs.pred_len
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.mul_season = MultiFourier([2], [24 * 4])  # , [ 24, 24*4])

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        # self.mlp = configs.mlp

        kernel_size = 25
        self.moving_avg = moving_avg(kernel_size, stride=1)

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())

            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

            self.prompt = configs.prompt

            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_token = self.tokenizer(text="Predict the future time step given the trend",return_tensors="pt").to(device)

            self.token_len = len(self.gpt2_token['input_ids'][0])

            try:
                self.pool = configs.pool
                if self.pool:
                    self.prompt_record_plot = {}
                    self.prompt_record_id = 0
                    self.diversify = True


            except:
                self.pool = False

            if self.pool:
                self.prompt_key_dict = nn.ParameterDict({})
                self.prompt_value_dict = nn.ParameterDict({})
                # self.summary_map = nn.Linear(self.token_len, 1)
                self.summary_map = nn.Linear(self.patch_num, 1)
                self.pool_size = 30
                self.top_k = 3
                self.prompt_len = 3
                self.token_len = self.prompt_len * self.top_k
                for i in range(self.pool_size):
                    prompt_shape = (self.prompt_len, 768)
                    key_shape = (768)
                    self.prompt_value_dict[f"prompt_value_{i}"] = nn.Parameter(torch.randn(prompt_shape))
                    self.prompt_key_dict[f"prompt_key_{i}"] = nn.Parameter(torch.randn(key_shape))

                self.prompt_record = {f"id_{i}": 0 for i in range(self.pool_size)}
                self.prompt_record_trend = {}

                self.diversify = True

        #self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.in_layer  = nn.Sequential(
                         nn.Linear(configs.patch_size, 4*configs.seq_len),
                         nn.ReLU(),
                         nn.Linear(4*configs.seq_len, configs.d_model)
                         )

        self.out_layer = nn.Linear(configs.d_model * (self.patch_num + self.token_len), configs.pred_len)

        self.prompt_layer = nn.Linear(configs.d_model, configs.d_model)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        config = LoraConfig(
            # task_type=TaskType.CAUSAL_LM, # causal language model
            r=8,
            lora_alpha=16,
            # target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="lora_only",  # bias, set to only lora layers to train
            # modules_to_save=["classifier"],
        )

        self.gpt2 = get_peft_model(self.gpt2, config)
        print_trainable_parameters(self.gpt2)

        for layer in (self.gpt2, self.prompt_layer, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

        self.num_nodes = configs.num_nodes
        self.rev_in = RevIn(num_features=self.num_nodes).to(device)

    def store_tensors_in_dict(self, original_x, original_trend, original_season, original_noise, trend_prompts,
                              season_prompts, noise_prompts):
        # Assuming prompts are lists of tuples
        self.prompt_record_id += 1
        for i in range(original_x.size(0)):
            self.prompt_record_plot[self.prompt_record_id + i] = {
                'original_x': original_x[i].tolist(),
                'original_trend': original_trend[i].tolist(),
                'original_season': original_season[i].tolist(),
                'original_noise': original_noise[i].tolist(),
                'trend_prompt': trend_prompts[i],
                'season_prompt': season_prompts[i],
                'noise_prompt': noise_prompts[i],
            }

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def select_prompt(self, summary, prompt_mask=None):
        prompt_key_matrix = torch.stack(tuple([self.prompt_key_dict[i] for i in self.prompt_key_dict.keys()]))
        prompt_norm = self.l2_normalize(prompt_key_matrix, dim=1)  # Pool_size, C
        summary_reshaped = summary.view(-1, self.patch_num)
        summary_mapped = self.summary_map(summary_reshaped)
        summary = summary_mapped.view(-1, 768)
        summary_embed_norm = self.l2_normalize(summary, dim=1)
        similarity = torch.matmul(summary_embed_norm, prompt_norm.t())
        if not prompt_mask == None:
            idx = prompt_mask
        else:
            topk_sim, idx = torch.topk(similarity, k=self.top_k, dim=1)
        if prompt_mask == None:
            count_of_keys = torch.bincount(torch.flatten(idx), minlength=15)
            for i in range(len(count_of_keys)):
                self.prompt_record[f"id_{i}"] += count_of_keys[i].item()

        prompt_value_matrix = torch.stack(tuple([self.prompt_value_dict[i] for i in self.prompt_value_dict.keys()]))
        batched_prompt_raw = prompt_value_matrix[idx].squeeze(1)
        batch_size, top_k, length, c = batched_prompt_raw.shape  # [16, 3, 5, 768]
        batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

        batched_key_norm = prompt_norm[idx]
        summary_embed_norm = summary_embed_norm.unsqueeze(1)
        sim = batched_key_norm * summary_embed_norm
        reduce_sim = torch.sum(sim) / summary.shape[0]

        # Return the sorted tuple of selected prompts along with batched_prompt and reduce_sim
        selected_prompts = [tuple(sorted(row)) for row in idx.tolist()]
        # print("reduce_sim: ", reduce_sim)

        return batched_prompt, reduce_sim, selected_prompts

    def get_norm(self, x, d='norm'):
        # if d == 'norm':
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        return x, means, stdev

    def get_patch(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)  # 4, 1, 420
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # 4,1, 64, 16
        x = rearrange(x, 'b m n p -> (b m) n p')  # 4, 64, 16

        return x

    def get_emb(self, x, tokens=None):
        [a, b, c] = x.shape

        if self.pool:
            prompt_x, reduce_sim, selected_prompts = self.select_prompt(x, prompt_mask=None)
            for selected_prompt in selected_prompts:
                self.prompt_record[selected_prompt] = self.prompt_record.get(selected_prompt, 0) + 1
            selected_prompts = selected_prompts
        else:
            prompt_x = self.gpt2.wte(tokens)
            prompt_x = prompt_x.repeat(a, 1, 1)
            prompt_x = self.prompt_layer(prompt_x)
        x = torch.cat((prompt_x, x), dim=1)

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        #if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec) # x_enc（4, 512, 1）
        return dec_out[:, -self.pred_len:, :] #（4,96，1）
        #return None

    def forecast(self, x, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x.shape  # 4, 512, 1
        x = self.rev_in(x, 'norm')

        x = self.get_patch(x)

        x = self.in_layer(x)  # 4, 64, 768

        trend = self.get_emb(x, self.gpt2_token['input_ids'])

        x = self.gpt2(inputs_embeds=trend).last_hidden_state

        trend = x[:, :self.token_len + self.patch_num, :]

        trend = self.out_layer(trend.reshape(B * M, -1))  # 4, 96
        trend = rearrange(trend, '(b m) l -> b l m', b=B)  # 4, 96, 1

        outputs = self.rev_in(trend, 'denorm')

        return outputs