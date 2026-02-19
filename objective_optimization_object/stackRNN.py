#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# ---- graceful fallback for smiles augmentation ----
try:
    from smiles_enumerator import SmilesEnumerator  # optional
except Exception:
    class SmilesEnumerator:
        def randomize_smiles(self, s: str) -> str:
            return s

class InvalidArgumentError(Exception):
    pass


class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_type='GRU',
                 n_layers=1, is_bidirectional=False, has_stack=False,
                 stack_width=None, stack_depth=None, use_cuda=None,
                 optimizer_instance=torch.optim.Adadelta, lr=0.01,
                 # ===== 新增：条件向量 z 注入 =====
                 cond_dim: int = 0, cond_scale: float = 1.0):
        super(StackAugmentedRNN, self).__init__()

        if layer_type not in ['GRU', 'LSTM']:
            raise InvalidArgumentError('Layer type must be GRU or LSTM')
        self.layer_type = layer_type
        self.is_bidirectional = is_bidirectional
        self.num_dir = 2 if self.is_bidirectional else 1
        self.has_cell = (layer_type == 'LSTM')
        self.has_stack = has_stack
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if self.has_stack:
            self.stack_width = stack_width
            self.stack_depth = stack_depth

        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.n_layers = n_layers

        # ===== 条件 z 的模块 =====
        self.cond_dim = int(cond_dim)
        self.cond_scale = float(cond_scale)
        self._cond_vec = None  # [1, cond_dim] on device or None
        if self.cond_dim > 0:
            # z -> 初始隐状态偏置（拼到每层每向）
            self.cond2h = nn.Linear(self.cond_dim, self.n_layers * self.num_dir * self.hidden_size, bias=False)
            # z -> 每步输入嵌入的偏置
            self.cond2in = nn.Linear(self.cond_dim, self.hidden_size, bias=False)

        if self.has_stack:
            self.stack_controls_layer = nn.Linear(self.hidden_size * self.num_dir, 3)
            self.stack_input_layer = nn.Linear(self.hidden_size * self.num_dir, self.stack_width)

        self.encoder = nn.Embedding(input_size, hidden_size)
        rnn_input_size = hidden_size + (stack_width if self.has_stack else 0)

        if self.layer_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, n_layers,
                               bidirectional=self.is_bidirectional)
        else:
            self.rnn = nn.GRU(rnn_input_size, hidden_size, n_layers,
                              bidirectional=self.is_bidirectional)
        self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        if self.use_cuda:
            self = self.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = self.optimizer_instance(self.parameters(), lr=lr, weight_decay=0.00001)

    # ======= 条件 z 的接口 =======
    def set_condition(self, z: torch.Tensor, scale: float = None):
        """
        z: [1, cond_dim] 或 [B=1, cond_dim]，只在 RL 里每个 episode 设置一次。
        """
        if self.cond_dim <= 0:
            return
        if z.dim() == 1:
            z = z.view(1, -1)
        if self.use_cuda:
            z = z.cuda()
        self._cond_vec = z.contiguous()
        if scale is not None:
            self.cond_scale = float(scale)

    def clear_condition(self):
        self._cond_vec = None

    def _cond_in_bias(self):
        if self.cond_dim <= 0 or self._cond_vec is None:
            return None
        b = self.cond2in(self._cond_vec) * self.cond_scale      # [1, H]
        return b

    def _cond_h_bias(self):
        if self.cond_dim <= 0 or self._cond_vec is None:
            return None
        b = self.cond2h(self._cond_vec) * self.cond_scale       # [1, L*D*H]
        b = b.view(self.n_layers * self.num_dir, 1, self.hidden_size)
        return b

    # ======= 常规方法 =======
    def load_model(self, path):
        weights = torch.load(path, map_location='cpu' if not self.use_cuda else None)
        self.load_state_dict(weights, strict=False)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def change_lr(self, new_lr):
        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr

    def forward(self, inp, hidden, stack):
        # inp: 一个 token（LongTensor 标量），这里先嵌入
        emb = self.encoder(inp.view(1, -1))   # [1,1,H]
        # 条件偏置加到输入嵌入
        if self.cond_dim > 0 and self._cond_vec is not None:
            b = self._cond_in_bias()  # [1,H]
            emb = emb + b.view(1, 1, -1)

        if self.has_stack:
            hidden_ = hidden[0] if self.has_cell else hidden
            hidden_2_stack = torch.cat((hidden_[0], hidden_[1]), dim=1) if self.is_bidirectional else hidden_.squeeze(0)
            stack_controls = F.softmax(self.stack_controls_layer(hidden_2_stack), dim=1)
            stack_input = torch.tanh(self.stack_input_layer(hidden_2_stack.unsqueeze(0)))
            stack = self.stack_augmentation(stack_input.permute(1, 0, 2), stack, stack_controls)
            stack_top = stack[:, 0, :].unsqueeze(0)
            emb = torch.cat((emb, stack_top), dim=2)

        output, next_hidden = self.rnn(emb.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, next_hidden, stack

    def stack_augmentation(self, input_val, prev_stack, controls):
        batch_size = prev_stack.size(0)
        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        zeros_at_the_bottom = Variable(zeros_at_the_bottom.cuda()) if self.use_cuda else Variable(zeros_at_the_bottom)
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack

    def init_hidden(self):
        t = torch.zeros(self.n_layers * self.num_dir, 1, self.hidden_size)
        h = Variable(t.cuda()) if self.use_cuda else Variable(t)
        # 条件偏置加到初始隐状态
        if self.cond_dim > 0 and self._cond_vec is not None:
            h = h + self._cond_h_bias()
        return h

    def init_cell(self):
        t = torch.zeros(self.n_layers * self.num_dir, 1, self.hidden_size)
        c = Variable(t.cuda()) if self.use_cuda else Variable(t)
        return c

    def init_stack(self):
        result = torch.zeros(1, self.stack_depth, self.stack_width)
        return Variable(result.cuda()) if self.use_cuda else Variable(result)

# stackRNN.py
    def train_step(self, inp, target, do_backward=True):
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        stack = self.init_stack() if self.has_stack else None
    
        loss = 0.0
        for c in range(len(inp)):
            output, hidden, stack = self(inp[c], hidden, stack)
            loss = loss + self.criterion(output, target[c].unsqueeze(0))
    
        loss = loss / max(1, len(inp))
        if do_backward:
            loss.backward()  # ← 这一步以前缺失
        return float(loss.item())


    @torch.no_grad()
    def evaluate(self, data, prime_str='<', end_token='>', predict_len=100):
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        stack = self.init_stack() if self.has_stack else None
    
        prime_input = data.char_tensor(prime_str)
        if self.use_cuda:                         # ← 新增
            prime_input = prime_input.cuda()      # ← 新增
        new_sample = prime_str
    
        for p in range(len(prime_str)-1):
            _, hidden, stack = self.forward(prime_input[p], hidden, stack)
        inp = prime_input[-1]
    
        for _ in range(predict_len):
            output, hidden, stack = self.forward(inp, hidden, stack)
            probs = torch.softmax(output, dim=1)
            top_i = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()
            predicted_char = data.all_characters[top_i]
            new_sample += predicted_char
            inp = data.char_tensor(predicted_char)
            if self.use_cuda:                     # ← 新增
                inp = inp.cuda()                  # ← 新增
            if predicted_char == end_token:
                break
        return new_sample

