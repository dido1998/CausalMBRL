import torch.nn as nn
import torch

from SCOFF.rnn_models_scoff import RNNModel as RNNModelScoff
from RIMs.rnn_models_rim import RNNModel as RNNModelRim


class RIM(nn.Module):
	def __init__(self, device, input_size, hidden_size, num_units, k, rnn_cell = 'LSTM', n_layers = 1, bidirectional = False, num_rules = 0, rule_time_steps = 0):
		super().__init__()
		"""
		- Wrapper for RIMs.
		- Mirrors nn.LSTM or nn.GRU
		- supports bidirection and multiple layers
		- Option to specify num_rules and rule_time_steps.

		Parameters:
			device: 'cuda' or 'cpu'
			input_size
			hidden_size
			num_units: Number of RIMs
			k: topk
			rnn_cell: 'LSTM' or 'GRU' (default = LSTM)
			n_layers: num layers (default = 1)
			bidirectional: True or False (default = False)
			num_rules: number of rules (default = 0)
			rule_time_steps: Number of times to apply rules per time step (default = 0)
		"""
		
		if device == 'cuda':
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.n_layers = n_layers
		self.num_directions = 2 if bidirectional else 1
		self.rnn_cell = rnn_cell
		self.num_units = num_units
		self.hidden_size = hidden_size // num_units
		if self.num_directions == 2:
			self.rimcell = nn.ModuleList([RNNModelRim(rnn_cell, input_size, input_size, [hidden_size], 1,  num_blocks = [num_units], topk = [k], do_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps).to(self.device) if i < 2 else 
				RNNModelRim(rnn_cell, 2 * hidden_size, 2 * hidden_size, [hidden_size], 1, num_blocks = [num_units], topk = [k], do_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps).to(self.device) for i in range(self.n_layers * self.num_directions)])
		else:
			self.rimcell = nn.ModuleList([RNNModelRim(rnn_cell, input_size, input_size, [hidden_size], 1,  num_blocks = [num_units], topk = [k], do_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps).to(self.device) if i == 0 else
				RNNModelRim(rnn_cell, hidden_size * num_units, hidden_size * num_units, [hidden_size], 1,  num_blocks = [num_units], topk = [k], do_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps).to(self.device) for i in range(self.n_layers)])

	def rim_transform_hidden(self, hs):
	    hiddens = []
	    h_split = torch.split(hs[0], 1, dim = 0)
	    c_split = torch.split(hs[1], 1, dim = 0)
	    for h, c in zip(h_split, c_split):
	        hiddens.append((h.squeeze(0), c.squeeze(0)))
	    return hiddens

	def rim_inverse_transform_hidden(self, hs):
	    h, c = [], []
	    for h_ in hs:
	        h.append(h_[0])
	        c.append(h_[1])
	    h = torch.stack(h, dim = 0)
	    c = torch.stack(c, dim = 0)

	    return (h, c)

	def layer(self, rim_layer, x, h, c = None, direction = 0):
		batch_size = x.size(1)
		xs = list(torch.split(x, 1, dim = 0))
		if direction == 1: xs.reverse()
		xs = torch.cat(xs, dim = 0)
		

		hidden = self.rim_transform_hidden((h, c))
		
		outputs, hidden, _, _, _ = rim_layer(xs, hidden)
		

		#hs = h.squeeze(0).view(batch_size, self.num_units, -1)
		#cs = None
		#if c is not None:
		#	cs = c.squeeze(0).view(batch_size, self.num_units, -1)
		#outputs = []

		#for x in xs:
		#	x = x.squeeze(0)
		#	hs, cs = rim_layer(x, hs, cs)
		#	outputs.append(hs.view(1, batch_size, -1))
		hs, cs = self.rim_inverse_transform_hidden(hidden)

		outputs = list(torch.split(outputs, 1, dim = 0))
		
		if direction == 1: outputs.reverse()
		outputs = torch.cat(outputs, dim = 0)
		
		if c is not None:
			return outputs, hs.view(batch_size, -1), cs.view(batch_size, -1)
		else:
			return outputs, hs.view(batch_size, -1)

	def forward(self, x, hidden = None):
		"""
		Input: x (seq_len, batch_size, input_size
			   hidden tuple[(num_layers * num_directions, batch_size, hidden_size)] (Optional)
		Output: outputs (batch_size, seqlen, hidden_size *  num_directions)
				hidden tuple[(num_layers * num_directions, batch_size, hidden_size)]
		"""
		
		h, c = None, None
		if hidden is not None:
			h, c = hidden[0], hidden[1]
		

		hs = torch.split(h, 1, 0) if h is not None else torch.split(torch.zeros(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
		hs = list(hs)
		cs = None
		if self.rnn_cell == 'LSTM':
			cs = torch.split(c, 1, 0) if c is not None else torch.split(torch.zeros(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
			cs = list(cs)
		else:
			cs = hs
		for n in range(self.n_layers):
			idx = n * self.num_directions
			x_fw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx])
			
			if self.num_directions == 2:
				idx = n * self.num_directions + 1
				x_bw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx], direction = 1)
				x = torch.cat((x_fw, x_bw), dim = 2)
			else:
				x = x_fw
		hs = torch.stack(hs, dim = 0)
		cs = torch.stack(cs, dim = 0)
		if self.rnn_cell == 'GRU':
			return x, (hs, )
		else:		
			return x, (hs, cs)

class SCOFF(nn.Module):
	def __init__(self, device, input_size, hidden_size, num_units, k, num_templates = 2, rnn_cell = 'LSTM', n_layers = 1, bidirectional = False, num_rules = 0, rule_time_steps = 0, perm_inv = False):
		super().__init__()
		"""
		- Wrappper for SCOFF.
		- Mirrors nn.LSTM or nn.GRU
		- supports bidirection and multiple layers
		- Option to specify num_rules and rule_time_steps.

		Parameters:
			device: 'cuda' or 'cpu'
			input_size
			hidden_size
			num_units: Number of RIMs
			k: topk
			rnn_cell: 'LSTM' or 'GRU' (default = LSTM)
			n_layers: num layers (default = 1)
			bidirectional: True or False (default = False)
			num_rules: number of rules (default = 0)
			rule_time_steps: Number of times to apply rules per time step (default = 0)
		"""
		if input_size % num_units != 0:
			print('ERROR: input_size should be evenly divisible by num_units')
			exit()
		if device == 'cuda':
			self.device = torch.device('cuda')
		else:
			self.device = torch.device('cpu')
		self.n_layers = n_layers
		self.num_directions = 2 if bidirectional else 1
		self.rnn_cell = rnn_cell
		self.num_units = num_units
		self.hidden_size = hidden_size // num_units
		if self.num_directions == 2:
			self.rimcell = nn.ModuleList([RNNModelScoff(rnn_cell, input_size, input_size, hidden_size, 1,  n_templates = num_templates, num_blocks = num_units, update_topk = k, use_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps, perm_inv = perm_inv).to(self.device) if i < 2 else 
				RNNModelScoff(rnn_cell, 2 * hidden_size, 2 * hidden_size, hidden_size, 1, n_templates = num_templates, num_blocks = num_units, update_topk = k, use_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps, perm_inv = perm_inv).to(self.device) for i in range(self.n_layers * self.num_directions)])
		else:
			self.rimcell = nn.ModuleList([RNNModelScoff(rnn_cell, input_size, input_size, hidden_size, 1, n_templates = num_templates, num_blocks = num_units, update_topk = k, use_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps, perm_inv = perm_inv).to(self.device) if i == 0 else
				RNNModelScoff(rnn_cell, hidden_size * num_units, hidden_size * num_units, hidden_size, 1, n_templates = num_templates,  num_blocks = num_units, update_topk = k, use_gru = rnn_cell == 'GRU', num_rules = num_rules, rule_time_steps = rule_time_steps, perm_inv = perm_inv).to(self.device) for i in range(self.n_layers)])

	def layer(self, rim_layer, x, h, c = None, direction = 0):
		batch_size = x.size(1)
		xs = list(torch.split(x, 1, dim = 0))
		if direction == 1: xs.reverse()
		xs = torch.cat(xs, dim = 0)
		

		hidden = (h, c)#self.rim_transform_hidden((h, c))
		
		outputs, hidden, _, _, _ = rim_layer(xs, hidden)
		

		#hs = h.squeeze(0).view(batch_size, self.num_units, -1)
		#cs = None
		#if c is not None:
		#	cs = c.squeeze(0).view(batch_size, self.num_units, -1)
		#outputs = []

		#for x in xs:
		#	x = x.squeeze(0)
		#	hs, cs = rim_layer(x, hs, cs)
		#	outputs.append(hs.view(1, batch_size, -1))
		hs, cs = hidden #self.rim_inverse_transform_hidden(hidden)

		outputs = list(torch.split(outputs, 1, dim = 0))
		
		if direction == 1: outputs.reverse()
		outputs = torch.cat(outputs, dim = 0)
		
		if c is not None:
			return outputs, hs.view(batch_size, -1), cs.view(batch_size, -1)
		else:
			return outputs, hs.view(batch_size, -1)

	def forward(self, x, hidden = None):
		"""
		Input: x (seq_len, batch_size, feature_size
			   hidden tuple[(num_layers * num_directions, batch_size, hidden_size * num_units)]
		Output: outputs (batch_size, seqlen, hidden_size * num_units * num-directions)
				h(and c) (num_layer * num_directions, batch_size, hidden_size* num_units)
		"""
		
		h, c = None, None
		if hidden is not None:
			h, c = hidden[0], hidden[1]
		

		hs = torch.split(h, 1, 0) if h is not None else torch.split(torch.zeros(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
		hs = list(hs)
		cs = None
		if self.rnn_cell == 'LSTM':
			cs = torch.split(c, 1, 0) if c is not None else torch.split(torch.zeros(self.n_layers * self.num_directions, x.size(1), self.hidden_size * self.num_units).to(self.device), 1, 0)
			cs = list(cs)
		else:
			cs = hs
		for n in range(self.n_layers):
			idx = n * self.num_directions
			x_fw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx])
			
			if self.num_directions == 2:
				idx = n * self.num_directions + 1
				x_bw, hs[idx], cs[idx] = self.layer(self.rimcell[idx], x, hs[idx], cs[idx], direction = 1)
				x = torch.cat((x_fw, x_bw), dim = 2)
			else:
				x = x_fw
		hs = torch.stack(hs, dim = 0)
		cs = torch.stack(cs, dim = 0)
		if self.rnn_cell == 'GRU':
			return x, (hs, hs)
		else:		
			return x, (hs, cs)



if __name__=='__main__':
	rim = SCOFF('cuda', 20, 32, 4, 4, rnn_cell = 'LSTM', n_layers = 2, bidirectional = True, num_rules = 5, rule_time_steps = 3, perm_inv = True)
	x = torch.rand(10, 2, 20).cuda()
	out = rim(x)
	print(out[0].size())
	print(out[1][0].size())
	print(out[1][1].size())