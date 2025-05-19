'''implement lstm'''
import torch
import torch.nn as nn

from expr.expression import *
import numpy as np
import math
from .utils import *

binary_code_len=4
nhead = 8

class TransformerActor(nn.Module):
    def __init__(self, opts, tokenizer):
        super().__init__()
        self.opts = opts
        self.max_layer = opts.max_layer
        self.output_size = tokenizer.vocab_size
        self.hidden_size = opts.hidden_dim
        self.num_layers = opts.num_layers
        self.max_c = opts.max_c
        self.min_c = opts.min_c
        self.fea_size = opts.fea_dim
        self.tokenizer = tokenizer
        self.interval = opts.c_interval
        ff_dim = self.hidden_size * 2
        # Transformer Encoder
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=opts.nhead, dim_feedforward=opts.ff_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=nhead, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output network
        self.output_net = nn.Linear(self.hidden_size, self.output_size)
        
        # Feature to hidden state projection
        if self.opts.fea_mode == 'xy':
            net_config = [{'in': self.fea_size, 'out': 16, 'drop_out': 0, 'activation': 'ReLU'},
                          {'in': 16, 'out': self.hidden_size, 'drop_out': 0, 'activation': 'None'}]
            self.x_to_c = MLP(net_config)
        else:
            self.x_to_c = nn.Linear(self.fea_size, self.hidden_size)
        
        # Constant value network
        self.constval_net = nn.Linear(self.hidden_size, int((self.max_c - self.min_c) // self.interval))
        self.num_c = int((self.max_c - self.min_c) // self.interval)

    def forward(self, x, save_data=False, fix_action=None):
        bs = x.shape[0]
        device = x.device

        log_prob_whole = torch.zeros(bs).to(device)
        
        # Initial input and memory for Transformer
        h_0 = torch.zeros((bs, 1, self.hidden_size)).to(device)
        if self.opts.fea_mode == 'xy':
            c_0 = self.x_to_c(x)
            c_0 = torch.mean(c_0, -2).unsqueeze(1)
        else:
            c_0 = self.x_to_c(x).unsqueeze(1)

        c=c_0
        h=h_0

        if save_data:
            memory = Memory()

        if not fix_action:
            x_in = torch.zeros((bs, 1, int(2**self.max_layer - 1) * binary_code_len)).to(device)
            transformer_memory = torch.cat([c, h], dim=1)
            len_seq = int(2**self.max_layer - 1)
            seq = torch.ones((bs, len_seq), dtype=torch.long) * -1
            const_vals = torch.zeros((bs, len_seq))

            position = torch.zeros((bs,), dtype=torch.long)
            working_index = torch.arange(bs)

            while working_index.shape[0] > 0:
                # Transformer forward
                output = self.transformer(transformer_memory)
                out = self.output_net(output[:, -1, :]).unsqueeze(1)
                # out = self.output_net(output)

                mask = get_mask(seq[working_index], self.tokenizer, position, self.max_layer)
                log_prob, choice, binary_code = get_choice(out, mask)

                # Get c if required
                c_index = self.tokenizer.is_consts(choice)
                if np.any(c_index):
                    out_c = self.constval_net(output[c_index, -1, :])
                    log_prob_c, c_val = get_c(out_c, self.min_c, self.interval)
                    log_prob_whole[working_index[c_index]] += log_prob_c
                    const_vals[working_index[c_index], position[c_index]] = c_val.cpu()

                # Update x_in with binary codes
                x_in = x_in.clone().detach()
                binary_code = binary_code.to(device)
                for i in range(binary_code_len):
                    x_in[range(len(working_index)), 0, position * binary_code_len + i] = binary_code[:, i]

                log_prob_whole[working_index] += log_prob
                seq[working_index, position] = choice.cpu()

                position = get_next_position(seq[working_index], choice, position, self.tokenizer)

                # Update working index and memory
                filter_index = (position != -1)
                working_index = working_index[filter_index]
                position = position[filter_index]
                x_in = x_in[filter_index]
                transformer_memory = transformer_memory[filter_index]
                
                if save_data:
                    memory.filter_index.append(filter_index)

            if self.opts.require_baseline:
                rand_seq, rand_c_seq = self.get_random_seq(bs)
                if not save_data:
                    return seq.numpy(), const_vals.numpy(), log_prob_whole, rand_seq, rand_c_seq
                else:
                    memory.seq = seq
                    memory.c_seq = const_vals
                    return seq.numpy(), const_vals.numpy(), log_prob_whole, rand_seq, rand_c_seq, memory.get_dict()

            if not save_data:
                return seq.numpy(), const_vals.numpy(), log_prob_whole
            else:
                memory.seq = seq
                memory.c_seq = const_vals
                return seq.numpy(), const_vals.numpy(), log_prob_whole, memory.get_dict()
        else:
            x_in=fix_action['x_in']     # x_in shape: (len, [bs,1,31*4])
            mask=fix_action['mask']     # mask shape: (len, [bs,vocab_size])
            working_index=fix_action['working_index']   # working_index
            # seq=torch.FloatTensor(fix_action['seq']).to(device)
            seq=fix_action['seq']
            c_seq=fix_action['c_seq']
            # c_seq=torch.FloatTensor(fix_action['c_seq']).to(device)
            position=fix_action['position']
            c_indexs=fix_action['c_index']
            filter_index=fix_action['filter_index']

            for i in range(len(x_in)):
                t_x_in = x_in[i]
                transformer_memory = torch.cat([c, h], dim=1)
                output = self.transformer(transformer_memory)
                w_index=working_index[i]
                pos=position[i]
                log_prob=get_choice(out,mask[i],fix_choice=seq[w_index,pos].to(device))
                log_prob_whole[w_index]+=log_prob

                c_index=c_indexs[i]
                # todo get c log_prob
                if np.any(c_index):
                    out_c=self.constval_net(output[c_index])
                    log_prob_c=get_c(out_c,self.min_c,self.interval,fix_c=c_seq[w_index[c_index],pos[c_index]])
                    log_prob_whole[w_index[c_index]]+=log_prob_c

                # update h & c
                h=h[:,filter_index[i]]
                c=c[:,filter_index[i]]
            return log_prob_whole
                
                
                
    def get_random_seq(self, bs):
        len_seq = int(2**self.max_layer - 1)
        seq = torch.ones((bs, len_seq), dtype=torch.long) * -1
        const_vals = torch.zeros((bs, len_seq))
        position = torch.zeros((bs,), dtype=torch.long)

        working_index = torch.arange(bs)
        while working_index.shape[0] > 0:
            output = torch.rand((working_index.shape[0], 1, self.output_size))
            mask = get_mask(seq[working_index], self.tokenizer, position, self.max_layer)

            _, choice, _ = get_choice(output, mask)
            c_index = self.tokenizer.is_consts(choice)

            if np.any(c_index):
                bs = output[c_index].shape[0]
                out_c = torch.rand((bs, 1, self.num_c))
                _, c_val = get_c(out_c, self.min_c, self.interval)
                const_vals[working_index[c_index], position[c_index]] = c_val

            seq[working_index, position] = choice
            position = get_next_position(seq[working_index], choice, position, self.tokenizer)

            filter_index = (position != -1)
            working_index = working_index[filter_index]
            position = position[filter_index]

        return seq.numpy(), const_vals.numpy()
    
if __name__ == '__main__':
    model=TransformerActor(fea_size=10,hidden_size=20,output_size=15,num_layers=1,max_length=10)
    x=torch.rand((8,10),dtype=torch.float)
    seq,log_prob=model(x)
    print(f'seq:{seq}')
    print(f'log_prob:{log_prob}')