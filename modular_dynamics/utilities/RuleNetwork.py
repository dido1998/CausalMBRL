import torch
import torch.nn as nn
import math
import numpy as np
from .GroupLinearLayer import GroupLinearLayer
from .attention_rim import MultiHeadAttention
import itertools

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

class GroupMLP(nn.Module):
	def __init__(self, in_dim, out_dim, num):
		super().__init__()
		self.group_mlp1 = GroupLinearLayer(in_dim, 128, num)
		self.group_mlp2 = GroupLinearLayer(128, out_dim, num)
		#self.group_mlp3 = GroupLinearLayer(128, 128, num)
		#self.group_mlp4 = GroupLinearLayer(128, out_dim, num)
		self.dropout = nn.Dropout(p = 0.5)


	def forward(self, x):
		x = torch.relu(self.dropout(self.group_mlp1(x)))
		x = torch.relu(self.dropout(self.group_mlp2(x)))
		#x = torch.relu(self.dropout(self.group_mlp3(x)))
		#x = torch.relu(self.dropout(self.group_mlp4(x)))
		return x

class MLP(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.mlp1 = nn.Linear(in_dim, 128)
		self.mlp2 = nn.Linear(128, 128)
		self.mlp3 = nn.Linear(128, 128)
		self.mlp4 = nn.Linear(128, out_dim)
		self.dropout = nn.Dropout(p = 0.5)

	def forward(self, x):
		x = torch.relu(self.dropout(self.mlp1(x)))
		x = torch.relu(self.dropout(self.mlp2(x)))
		x = torch.relu(self.dropout(self.mlp3(x)))
		x = torch.relu(self.dropout(self.mlp4(x)))
		return x


class RuleNetwork(nn.Module):
	def __init__(self, hidden_dim, num_variables,  num_rules = 4, rule_dim = 64, query_dim = 32, value_dim = 64, key_dim = 32, num_heads = 4, dropout = 0.1, design_config = None):
		super().__init__()
		self.rule_dim = rule_dim
		self.num_heads = num_heads
		self.key_dim = key_dim
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu ')
		self.value_dim = value_dim
		self.query_dim = query_dim
		self.hidden_dim = hidden_dim
		self.design_config = design_config

		self.rule_activation = []
		self.softmax = []
		self.masks = []
		self.rule_embeddings = nn.Parameter(0.01 * torch.randn(1, num_rules, rule_dim).to(self.device))
		self.transform_src = nn.Linear(300, 60)
		
		self.dropout = nn.Dropout(p = 0.5)

		self.positional_encoding = PositionalEncoding(hidden_dim)

		self.transform_rule = nn.Linear(rule_dim, hidden_dim)

		self.transformer_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = num_heads, dropout = 0.5)

		self.transformer = nn.TransformerEncoder(self.transformer_layer, 3)

		self.multihead_attention = nn.MultiheadAttention(hidden_dim, 4)
		
		self.variable_rule_select = MultiHeadAttention(n_head=4, d_model_read= hidden_dim, d_model_write = rule_dim , d_model_out = hidden_dim,  d_k=64, d_v=64, num_blocks_read = 2 * num_variables, num_blocks_write = num_rules, topk = num_rules, grad_sparse = False)

		self.encoder_transform = nn.Linear(num_variables * hidden_dim, hidden_dim)
		self.rule_mlp = GroupMLP(rule_dim + hidden_dim, hidden_dim, num_rules)
		self.rule_linear = GroupLinearLayer(rule_dim + hidden_dim, hidden_dim, num_rules)
		self.variables_select = MultiHeadAttention(n_head=4, d_model_read= hidden_dim, d_model_write = hidden_dim , d_model_out = hidden_dim,  d_k=64, d_v=64, num_blocks_read = 1, num_blocks_write = num_variables, topk = 3, grad_sparse = False)

		self.phase_1_mha = MultiHeadAttention(n_head = 1, d_model_read = 2 * hidden_dim * num_variables, d_model_write = hidden_dim, d_model_out = hidden_dim, d_k = 64, d_v = 64, num_blocks_read = 1, num_blocks_write = num_rules, topk = num_rules, grad_sparse = False)

		self.variable_mlp = MLP(2 * hidden_dim, hidden_dim)
		num = [i for i in range(num_variables)]
		num_comb = len(list(itertools.combinations(num, r = 2)))
		self.phase_2_mha = MultiHeadAttention(n_head = 1, d_model_read = hidden_dim, d_model_write = hidden_dim, d_model_out = hidden_dim, d_k = 64, d_v = 64, num_blocks_read = num_comb, num_blocks_write = 1, topk = 1, grad_sparse = False )
		self.variable_mlp_2 = GroupMLP(3 * hidden_dim, hidden_dim, num_variables)
	def transpose_for_scores(self, x, num_attention_heads, attention_head_size):
	    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
	    x = x.view(*new_x_shape)
	    return x.permute(0, 2, 1, 3)

	def forward(self, hidden, *args):
		if not self.design_config['grad']:
			hidden = hidden.detach()
		for arg in args:
			assert type(arg) == torch.Tensor and arg.dim() == 3			
		
		batch_size, num_variables, variable_dim = hidden.size()

		num_rules = self.rule_embeddings.size(1)
		rule_emb_orig = self.rule_embeddings.repeat(batch_size, 1, 1)
		#print(rule_emb)
		rule_emb = self.dropout(self.transform_rule(rule_emb_orig))
		
		if len(args) > 0:
			extra_input  = torch.cat(args, dim = 1).detach()
			#extra_input = self.transform_src(extra_input)
			start_index = [0]

			start_index.append(extra_input.size(1))
			start_index.append(start_index[-1] + num_variables)
			#extra_input = self.encoder_transform(extra_input)
			if self.design_config['transformer']:	
				transformer_input = torch.cat((extra_input, hidden, rule_emb), dim = 1)
			else:
				read_input = torch.cat((extra_input, hidden), dim = 1)
		else:
			start_index = [0, num_variables]
			transformer_input = torch.cat((hidden, rule_emb), dim = 1)

		if self.design_config['transformer']:
			transformer_input = transformer_input.transpose(0, 1)
			transformer_input = self.positional_encoding(transformer_input)
			transformer_out = self.transformer(transformer_input)
			attn_output, attn_output_weights = self.multihead_attention(transformer_out, transformer_out, transformer_out)
			transformer_out  = transformer_out.transpose(0, 1)
			variable_rule = attn_output_weights[:, start_index[-2]:start_index[-2] + num_variables,  start_index[-1]:]
			rule_variable = attn_output_weights[:,  start_index[-1]:, start_index[-2]: start_index[-2] + num_variables].transpose(1, 2)

			scores = variable_rule + rule_variable
			transformer_out  = transformer_out.transpose(0, 1)
		else:
			_, variable_rule, _ = self.variable_rule_select(read_input, rule_emb_orig, rule_emb_orig)
			scores = variable_rule[:, num_variables:, :]
			scores = torch.mean(scores.view(rule_emb_orig.size(0), -1, num_variables, num_rules), dim = 1)
		mask = torch.nn.functional.gumbel_softmax(scores.view(batch_size, -1), dim = 1, tau = 0.5, hard = True).view(batch_size, num_variables, num_rules)
		if self.design_config['application_option'] == 0:
			transformer_out = transformer_out.transpose(0, 1)
			scores = scores * mask
			value = transformer_out[:, start_index[-1]:, :]
			rule_mlp_output = torch.matmul(scores, value)
			return rule_mlp_output
		elif self.design_config['application_option'] == 1:
			scores = scores * mask
			value = rule_emb
			rule_mlp_output = torch.matmul(scores, value)
			return rule_mlp_output
		elif self.design_config['application_option'] == 2:
			variable_mask = torch.sum(mask, dim = 2).unsqueeze(-1)
			rule_mask = torch.sum(mask, dim = 1).unsqueeze(-1)

			selected_variable = torch.sum(hidden * variable_mask, dim = 1).unsqueeze(1).repeat(1, mask.size(2), 1)
			rule_mlp_input = torch.cat((rule_emb_orig, selected_variable), dim = 2)
			rule_mlp_output = self.rule_linear(rule_mlp_input)
			rule_mlp_output = torch.sum(rule_mlp_output * rule_mask, dim = 1).unsqueeze(1)

			relevant_variables, _, _ = self.variables_select(rule_mlp_output, hidden, hidden)
			rule_mlp_output = rule_mlp_output + relevant_variables
			rule_mlp_output = rule_mlp_output.repeat(1, hidden.size(1), 1)
			rule_mlp_output = rule_mlp_output * variable_mask
			return rule_mlp_output
		elif self.design_config['application_option'] == 3:
			variable_mask = torch.sum(mask, dim = 2).unsqueeze(-1)
			rule_mask = torch.sum(mask, dim = 1).unsqueeze(-1)

			selected_variable = torch.sum(hidden * variable_mask, dim = 1).unsqueeze(1).repeat(1, mask.size(2), 1)
			rule_mlp_input = torch.cat((rule_emb_orig, selected_variable), dim = 2)
			rule_mlp_output = self.rule_mlp(rule_mlp_input)
			rule_mlp_output = torch.sum(rule_mlp_output * rule_mask, dim = 1).unsqueeze(1)

			relevant_variables, _, _ = self.variables_select(rule_mlp_output, hidden, hidden)
			rule_mlp_output = rule_mlp_output + relevant_variables
			rule_mlp_output = rule_mlp_output.repeat(1, hidden.size(1), 1)
			rule_mlp_output = rule_mlp_output * variable_mask
			return rule_mlp_output
		elif self.design_config['application_option'] == 4:
			transformer_out = transformer_out.transpose(0, 1)
			scores = scores * mask
			value = transformer_out[:, start_index[-1]:, :]
			rule_mlp_output_1 = torch.matmul(scores, value)
			variable_mask = torch.sum(mask, dim = 2).unsqueeze(-1)
			rule_mask = torch.sum(mask, dim = 1).unsqueeze(-1)

			selected_variable = torch.sum(hidden * variable_mask, dim = 1).unsqueeze(1).repeat(1, mask.size(2), 1)
			rule_mlp_input = torch.cat((rule_emb_orig, selected_variable), dim = 2)
			rule_mlp_output = self.rule_mlp(rule_mlp_input)
			rule_mlp_output = torch.sum(rule_mlp_output * rule_mask, dim = 1).unsqueeze(1)

			relevant_variables, _, _ = self.variables_select(rule_mlp_output, hidden, hidden)
			rule_mlp_output = rule_mlp_output + relevant_variables
			rule_mlp_output = rule_mlp_output.repeat(1, hidden.size(1), 1)
			rule_mlp_output = rule_mlp_output * variable_mask
			return rule_mlp_output + rule_mlp_output_1
		else:
			_, phase_1_attn, _ = self.phase_1_mha(read_input.view(read_input.size(0), -1).unsqueeze(1), rule_emb, rule_emb)
			phase_1_attn = phase_1_attn.squeeze(1)
			
			mask = torch.nn.functional.gumbel_softmax(phase_1_attn, dim = 1, tau = 0.5, hard = True)
			mask = mask.unsqueeze(-1)
			rule_emb = rule_emb * mask
			rule_emb = torch.sum(rule_emb, dim = 1)
			variable_indices = torch.arange(0, num_variables).to(rule_emb.device)

			variable_indices = torch.combinations(variable_indices, r = 2)
			hidden_ = hidden.repeat(1, variable_indices.size(0), 1)
			aux_ind = np.arange(0, variable_indices.size(0))
			aux_ind = np.repeat(aux_ind, 2)
			aux_ind = torch.from_numpy(aux_ind * num_variables).to(rule_emb.device)
			variable_indices_ = variable_indices.view(-1) + aux_ind
			hidden_ = hidden_[:, variable_indices_, :]

			hidden_ = hidden_.view(hidden_.size(0), -1)
			hidden_ = torch.split(hidden_, 2 * variable_dim, dim = 1)
			hidden_ = torch.cat(hidden_, dim = 0)
			hidden_ = self.variable_mlp(hidden_)
			hidden_ = torch.split(hidden_, batch_size, dim = 0)
			hidden_ = torch.stack(hidden_, dim = 1)
			
			_, variable_attn, _ = self.phase_2_mha(hidden_, rule_emb.unsqueeze(1), rule_emb.unsqueeze(1))
			
			variable_attn = variable_attn.squeeze(-1)
			mask_variable = torch.nn.functional.gumbel_softmax(variable_attn, dim = 1, hard = True, tau = 0.5).unsqueeze(-1)
			
			
			hidden_ = hidden_ * mask_variable
			hidden_ = torch.sum(hidden_, dim = 1)
			mask_variable_argmax = torch.argmax(mask_variable.squeeze(2), dim = 1)
			selected_variable_indices = variable_indices[mask_variable_argmax]
			original_variable_mask = torch.zeros(hidden.size(0), hidden.size(1)).to(hidden.device)
			
			original_variable_mask.scatter_(1, selected_variable_indices, 1)
			original_variable_mask = original_variable_mask.unsqueeze(-1)
			hidden_ = hidden_.unsqueeze(1).repeat(1, hidden.size(1), 1)
			rule_emb = rule_emb.unsqueeze(1).repeat(1, hidden.size(1), 1)
			penultimate_representation = torch.cat((hidden, hidden_, rule_emb), dim = 2)
			final_representation = self.variable_mlp_2(penultimate_representation) * original_variable_mask
			return final_representation

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
