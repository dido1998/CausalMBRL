import torch
import torch.nn as nn
import math
import numpy as np
from .GroupLinearLayer import GroupLinearLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class blocked_grad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0


class RuleNetwork(nn.Module):
	def __init__(self, hidden_dim, num_variables,  num_rules = 4, rule_dim = 64, query_dim = 32, value_dim = 64, key_dim = 32, num_heads = 4, dropout = 0.1, input_size = 300):
		super().__init__()
		self.rule_dim = rule_dim
		self.num_heads = num_heads
		self.key_dim = key_dim
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu ')
		self.value_dim = value_dim
		self.query_dim = query_dim
		self.hidden_dim = hidden_dim
		self.input_size = input_size // 2

		self.rule_activation = []
		self.softmax = []
		self.masks = []

		self.rule_embeddings = nn.Parameter(torch.zeros(1, num_rules, rule_dim).to(self.device))
		"""self.query_layer = GroupLinearLayer(hidden_dim, query_dim * num_heads, num_variables)
		self.key_layer = nn.Linear(rule_dim, key_dim * num_heads)
		self.value_layer = nn.Linear(rule_dim, hidden_dim * num_heads)
		self.transform_hidden = GroupLinearLayer(hidden_dim + value_dim, hidden_dim, num_variables)"""
		self.dropout = nn.Dropout(p = dropout)

		self.positional_encoding = PositionalEncoding(hidden_dim)
		self.transform_rule = nn.Linear(rule_dim, hidden_dim)

		self.transformer_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = num_heads)
		self.transformer = nn.TransformerEncoder(self.transformer_layer, 3)
		self.multihead_attention = nn.MultiheadAttention(hidden_dim, 4)
		self.context_command_transform = nn.Linear(500, self.hidden_dim)
		self.context_situation_transform = nn.Linear(500, self.hidden_dim)
		self.value = nn.Linear(rule_dim, self.hidden_dim)

	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
	    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
	    x = x.view(*new_x_shape)
	    return x.permute(0, 2, 1, 3)

	def forward(self, hidden):
		"""query = self.query_layer(hidden)
		null_rule = torch.zeros(1, 1, self.rule_dim).to(self.device)
		#print(self.rule_embeddings)
		#rule_emb = torch.cat(self.rule_embeddings), dim = 1)
		key = self.key_layer(self.rule_embeddings)
		value = self.value_layer(self.rule_embeddings)
		query = self.transpose_for_scores(query, self.num_heads, self.query_dim)
		key = self.transpose_for_scores(key, self.num_heads, self.key_dim)

		value = torch.mean(self.transpose_for_scores(value, self.num_heads, self.hidden_dim), dim = 1)


		attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.key_dim) 
		attention_scores = self.dropout(torch.mean(attention_scores, dim = 1))
		batch_size, num_variables, num_rules = attention_scores.size()
		mask = torch.nn.functional.gumbel_softmax(attention_scores.view(batch_size, -1), tau = 0.5, dim =1, hard = True).view(batch_size, num_variables, num_rules)
		#null_attention_probs = torch.softmax(attention_scores, dim = 2)[:, :, 0]
		#min_null_weight = torch.argmin(null_attention_probs, dim = 1).detach()
		#variable_selection_mask = torch.zeros_like(null_attention_probs)
		#variable_selection_indices = np.arange(variable_selection_mask.size(0))
		#variable_selection_mask[variable_selection_indices, min_null_weight] = 1.
		#variable_selection_mask = variable_selection_mask.unsqueeze(-1)
		
		mask = torch.nn.functional.gumbel_softmax(attention_scores[:, :, 1:], tau = 1, dim = 2, hard = True)
		mask = mask * variable_selection_mask
		mask = torch.cat(((1 - variable_selection_mask),mask), dim = 2)

		mask_ = torch.sum(mask, dim = 2)
		
		self.rule_activation.append(torch.argmax(mask, dim = 2).cpu().numpy())
		self.softmax.append(attention_scores.detach().cpu().numpy())
		
		attention_scores = mask * torch.softmax(attention_scores, dim = 2)
		
		value = torch.matmul(attention_scores, value)		

		#mask = mask[:, :, 1:]
		#mask = torch.sum(mask, dim = -1).unsqueeze(-1)

		#mask = mask.detach()

		mask_ = mask_.unsqueeze(-1)
		hidden = mask_ * (value + hidden) + (1 - mask_) * hidden
		return hidden"""

		
		

		
		batch_size, num_variables, variable_dim = hidden.size()

		num_rules = self.rule_embeddings.size(1)
		rule_emb = self.rule_embeddings.repeat(batch_size, 1, 1)
		rule_emb = self.dropout(self.transform_rule(rule_emb))
		zero_rule = torch.zeros(batch_size, 1, rule_emb.size(-1), device = rule_emb.device)
		rule_emb = torch.cat((zero_rule, rule_emb), dim = 1)

		
		if False:
			args_ = []
			context_command = self.context_command_transform(context_command)
			context_situation = self.context_situation_transform(context_situation)
			extra_input  = torch.cat((context_command, context_situation), dim = 1).detach()
			#extra_input = torch.cat((extra_input), dim = 1)
			start_index = [0]

			start_index.append(extra_input.size(1))
			start_index.append(start_index[-1] + num_variables)
			#extra_input = self.encoder_transform(extra_input)	
			transformer_input = torch.cat((extra_input, hidden.detach(), rule_emb.detach()), dim = 1)
		else:
			start_index = [0, num_variables]
			transformer_input = torch.cat((hidden.detach(), rule_emb), dim = 1)

		transformer_input = transformer_input.transpose(0, 1)
		transformer_input = self.positional_encoding(transformer_input)
		transformer_out = self.transformer(transformer_input)
		attn_output, attn_output_weights = self.multihead_attention(transformer_out, transformer_out, transformer_out)

		variable_rule = attn_output_weights[:, start_index[-2]:start_index[-2] + num_variables,  start_index[-1]:]
		rule_variable = attn_output_weights[:,  start_index[-1]:, start_index[-2]: start_index[-2] + num_variables].transpose(1, 2)


		scores = variable_rule + rule_variable
		
		mask = torch.nn.functional.gumbel_softmax(scores.view(batch_size, -1), dim = 1, tau = 0.5, hard = True).view(batch_size, num_variables, num_rules + 1)
		mask_no_rule = mask[:, :, 0:1]
		mask_block_grad = torch.sum(mask, dim = 2).unsqueeze(-1)
		#hidden = blocked_grad.apply(hidden, mask_block_grad)


		self.rule_activation.append(torch.argmax(mask, dim = 2).cpu().numpy())
		self.softmax.append(scores.detach().cpu().numpy())
		scores = scores * mask
		transformer_out  = transformer_out.transpose(0, 1)
		#print(rule_emb.size())
		mask_block_grad2 = torch.sum(mask, dim = 1).unsqueeze(-1)
		rule_emb = blocked_grad.apply(rule_emb, mask_block_grad2)
		value = transformer_out[:, start_index[-1]:, :]#rule_emb
		value = torch.matmul(scores, value)
		return mask_no_rule * hidden + (1 - mask_no_rule) * (hidden + value)

	def reset_activations(self):
		self.rule_activation = []
		self.softmax = []

if __name__ == '__main__':
	model = RuleNetwork(6, 4).cuda()


	hiddens = torch.autograd.Variable(torch.randn(3, 4, 6), requires_grad = True).cuda()
	new_hiddens = model(hiddens)

	
	hiddens.retain_grad()
	new_hiddens.backward(torch.ones(hiddens.size()).cuda())
	
	#print(model.rule_embeddings.grad)
	#print(model.query_layer.w.grad)
