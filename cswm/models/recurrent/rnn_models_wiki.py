import torch.nn as nn
import torch
from .attention import MultiHeadAttention, Seq2SeqAttention
from .layer_conn_attention import LayerConnAttention
from .BlockLSTM import BlockLSTM
import random
import time
from .GroupLinearLayer import GroupLinearLayer
from .sparse_grad_attn import blocked_grad
import numpy as np

from .blocks_core import BlocksCore

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, use_cudnn_version=True,
                 use_adaptive_softmax=False, cutoffs=None, discrete_input=False, num_blocks=[6], topk=[4], do_gru=False,
                 use_inactive=False, blocked_grad=False, layer_dilation = -1, block_dilation = -1, num_modules_read_input=2, use_linear = False, is_decoder = False, use_attention = False):

        super(RNNModel, self).__init__()

        self.topk = topk
        print('Top k Blocks: ', topk)

        self.use_cudnn_version = use_cudnn_version
        self.drop = nn.Dropout(dropout)

        print('Number of Inputs, ninp: ', ninp)
        if discrete_input:
            self.encoder = nn.Embedding(ntoken, ninp)
        else:
            self.encoder = nn.Linear(ntoken, ninp)

        self.num_blocks = num_blocks
        print('Number of Blocks: ', self.num_blocks)

        self.nhid = nhid
        print('Dimensions of Hidden Layers: ', nhid)

        self.discrete_input = discrete_input
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(dim=1)

        self.use_inactive = use_inactive
        self.blocked_grad = blocked_grad
        print('Is the model using inactive blocks for higher representations? ', use_inactive)

        if layer_dilation == -1:
            self.layer_dilation = [1]*nlayers
        else:
            self.layer_dilation = layer_dilation

        if block_dilation == -1:
            self.block_dilation = [1]*nlayers
        else:
            self.block_dilation = block_dilation

        num_blocks_in = [1 for i in topk]

        self.bc_lst = []
        self.dropout_lst = []

        print("Dropout rate", dropout)
        self.use_attention = use_attention

        if is_decoder and use_attention:
            print(nhid)
            self.attention = Seq2SeqAttention(nhid[0], nhid[0])

        for i in range(nlayers):
            if i==0:
                if is_decoder and use_attention:
                    self.bc_lst.append(BlocksCore(nhid[i] + ninp,nhid[i], num_blocks_in[i], num_blocks[i], topk[i], True, do_gru=do_gru, num_modules_read_input=num_modules_read_input))
                else:
                    self.bc_lst.append(BlocksCore(ninp,nhid[i], num_blocks_in[i], num_blocks[i], topk[i], True, do_gru=do_gru, num_modules_read_input=num_modules_read_input))
            else:
                self.bc_lst.append(BlocksCore(nhid[i-1],nhid[i], num_blocks_in[i], num_blocks[i], topk[i], True, do_gru=do_gru, num_modules_read_input=num_modules_read_input))
        for i in range(nlayers - 1):
            self.dropout_lst.append(nn.Dropout(dropout))

        self.bc_lst = nn.ModuleList(self.bc_lst)
        self.dropout_lst = nn.ModuleList(self.dropout_lst)

        self.use_linear = use_linear
        self.use_adaptive_softmax = use_adaptive_softmax
        if use_linear:
            self.decoder = nn.Linear(nhid[-1] + ninp, ntoken)
            if tie_weights:
            	if nhid[-1] != ninp:
            		raise ValueError('When using the tied flag, nhid must be equal to emsize')
            	else:
            		self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.is_decoder = is_decoder

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if not self.use_adaptive_softmax and self.use_linear:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, calc_mask=False):
        extra_loss = 0.0

        emb = input#self.drop(self.encoder(input))
        """weighted = None
        encoder_inputs_ = None
        attn_vec = None
        if self.is_decoder and self.use_attention:
            a = self.attention(hidden[0][0], encoder_outputs)
                
            #a = [batch size, src len]

            #max_indices = torch.argmax(a, dim = 1).detach()
            #mask = torch.zeros(a.size(0), a.size(1), 1,).to(a.device).detach()
            #x_ind = [i for i in range(a.size(0))]

            #mask[x_ind, max_indices] = 1
            #mask = mask.transpose(0, 1)
            #encoder_inputs = encoder_inputs * mask
            #encoder_inputs = torch.sum(encoder_inputs, dim = 0)
            #encoder_inputs = encoder_inputs.unsqueeze(1)
            
            #mask_2 = torch.zeros_like(a).detach()

            #topk = torch.topk(a, dim = 1, k = 3)

            #x_ind = np.repeat(np.array(x_ind), 3)

            #mask_2[x_ind, topk.indices.view(-1)] = 1
            #a = a * mask_2

            a = a.unsqueeze(1)
            
            #a = [batch size, 1, src len]
            
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            
            #encoder_outputs = [batch size, src len, enc hid dim * 2]
            encoder_inputs = encoder_inputs.permute(1, 0, 2)
            encoder_inputs = torch.bmm(a, encoder_inputs)
            encoder_inputs = encoder_inputs.permute(1, 0, 2)
            encoder_inputs_ = encoder_inputs.permute(1, 0, 2)

            
            weighted = torch.bmm(a, encoder_outputs)
            
            #weighted = [batch size, 1, enc hid dim * 2]
            
            weighted = weighted.permute(1, 0, 2)
            
            #weighted = [1, batch size, enc hid dim * 2]
        
            emb = torch.cat((emb, weighted), dim = 2)
            #emb = weighted

            attn_vec = weighted.view(hidden[0][0].size())

            #encoder_final_state = weighted.squeeze(0)"""
        if True:
            # for loop implementation with RNNCell
            layer_input = emb
            new_hidden = [[] for _ in range(self.nlayers)]
            if calc_mask:
                masks = [[] for _ in range(self.nlayers)]
                sample_masks = [[] for _ in range(self.nlayers)]
            for idx_layer in range(self.nlayers):
                
                output = []
                t0 = time.time()
                self.bc_lst[idx_layer].blockify_params()
                hx, cx = hidden[int(idx_layer)][0], hidden[int(idx_layer)][1]
                for idx_step in range(input.shape[0]):
                    if idx_step % self.layer_dilation[idx_layer] == 0:
                        if idx_step % self.block_dilation[idx_layer] == 0:
                            hx, cx, mask = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx, idx_step, do_block = True)
                        else:
                            hx, cx, mask = self.bc_lst[idx_layer](layer_input[idx_step], hx, cx, idx_step, do_block = False)

                    if idx_layer < self.nlayers - 1:
                        if self.use_inactive:
                            if self.blocked_grad:
                                bg = blocked_grad()
                                output.append(bg(hx,mask))
                            else:
                                output.append(hx)
                        else:
                            if self.blocked_grad:
                                bg = blocked_grad()
                                output.append((mask)*bg(hx,mask))
                            else:
                                output.append((mask)*hx)
                    else:
                        output.append(hx)

                    if calc_mask:
                        mk = mask.view(mask.size()[0], self.num_blocks[idx_layer], self.nhid[idx_layer] // self.num_blocks[idx_layer])
                        mk = torch.mean(mk, dim=2)
                        sample_masks[idx_layer].append(mk[0])
                        mk = torch.mean(mk, dim=0)
                        masks[idx_layer].append(mk)

                if calc_mask:
                    masks[idx_layer] = torch.stack(masks[idx_layer]).view(input.shape[0],self.num_blocks[idx_layer]).transpose(1,0)
                    sample_masks[idx_layer] = torch.stack(sample_masks[idx_layer]).view(input.shape[0],self.num_blocks[idx_layer]).transpose(1,0)

                output = torch.stack(output)

                if idx_layer < self.nlayers - 1:
                    layer_input = self.dropout_lst[idx_layer](output)
                else:
                    layer_input = output

                new_hidden[idx_layer] = tuple((hx,cx))

            hidden = new_hidden
        

        output = self.drop(output)
        if self.use_linear:
            output = torch.cat((output, encoder_inputs), dim = 2)
            dec = output.view(output.size(0) * output.size(1), self.nhid[-1] + self.ninp)
        else:
            dec = output.view(output.size(0) * output.size(1), self.nhid[-1])
        if self.use_linear:
            dec = self.decoder(dec)
        if calc_mask:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden, emb, extra_loss, masks, sample_masks, None
        else:
            return dec.view(output.size(0), output.size(1), dec.size(1)), hidden, emb, extra_loss, None, None, None


    def init_hidden(self, bsz):
        weight = next(self.bc_lst[0].block_lstm.parameters())
        hidden = []
        if True or self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            for i in range(self.nlayers):
                hidden.append((weight.new_zeros(bsz, self.nhid[i]),
                    weight.new_zeros(bsz, self.nhid[i])))
            # return (weight.new_zeros(self.nlayers, bsz, self.nhid),
            #         weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            for i in range(self.nlayers):
                hidden.append((weight.new_zeros(bsz, self.nhid[i])))

        return hidden
