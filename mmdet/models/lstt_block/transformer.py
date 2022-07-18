import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from .basic import DropPath, GroupNorm1D, GNActGCTDWConv2d, seq_to_2d
from .gct import GCT


def _get_norm(indim, type='ln', groups=8):
    if type == 'gn':
        return GroupNorm1D(indim, groups)
    else:
        return nn.LayerNorm(indim)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gele/glu, not {activation}.")


class LongShortTermTransformer(nn.Module):

    def __init__(self,
                 num_layers=2,
                 d_model=256,
                 self_nhead=4,
                 att_nhead=2,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="gelu",
                 return_intermediate=False,
                 final_norm=True):

        super().__init__()
        layers = []

        for idx in range(num_layers):
            layers.append(LongShortTermTransformerBlock(d_model, self_nhead,
                                                        att_nhead, dim_feedforward,
                                                        dropout, activation,
                                                        drop_long_term=False if idx == 0 else True))

        num_norms = num_layers
        if not final_norm:
            num_norms -= 1
        self.decoder_norms = [_get_norm(d_model, type='ln') for _ in range(
            num_layers)] if num_norms > 0 else None
        self.final_norm = final_norm

        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers
        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)
        self.return_intermediate = return_intermediate

        self._init_weight()

    def forward(self,
                tgt,
                long_term_memories,
                short_term_memories,
                long_term_id=None,
                self_pos=None,
                size_2d=None):

        output = tgt

        intermediate = []
        intermediate_embs = []

        for idx, layer in enumerate(self.layers):
            output, embs = layer(output, long_term_memories[idx] if long_term_memories is not None else None,
                                 short_term_memories[idx] if short_term_memories is not None else None,
                                 long_term_id=long_term_id, self_pos=self_pos, size_2d=size_2d)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_embs.append(embs)

        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

                for idx in range(len(intermediate) - 1):
                    intermediate[idx] = self.decoder_norms[idx](
                        intermediate[idx])

        if self.return_intermediate:
            return intermediate, intermediate_embs

        return output, embs

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LongShortTermTransformerBlock(nn.Module):

    def __init__(self, d_model, self_nhead, att_nhead, dim_feedforward=1024, dropout=0.1,
                 activation="gelu", local_dilation=1, drop_long_term=True):
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, self_nhead)

        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)

        self.long_term_attn = MultiheadAttention(
            d_model, att_nhead, use_linear=False)
       # self.short_term_attn = MultiheadLocalAttention(
       #     d_model, att_nhead, dilation=local_dilation, use_linear=False)
        self.short_term_attn = MultiheadLocalAttentionV2(
            d_model, att_nhead, dilation=local_dilation, use_linear=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)

        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = _get_norm(d_model)
        self.norm2 = _get_norm(d_model)
        self.norm3 = _get_norm(d_model)

        self.droppath = DropPath(dropout)
        self.drop_long_term = drop_long_term

        self.activation = GNActGCTDWConv2d(dim_feedforward)

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                long_term_memory,
                short_term_memory,
                long_term_id=None,
                self_pos=None,
                size_2d=(30, 30)):

        # Self-attention
        _tgt = self.norm2(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]
        tgt2 = self.droppath(tgt2)

        tgt = tgt + tgt2

        # Long Short-Term Attention
        _tgt = self.norm1(tgt)

        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt

        local_Q = seq_to_2d(curr_Q, size_2d)

        if long_term_memory is None:
            global_K = curr_K
            global_V = self.linear_V(_tgt + long_term_id)
            local_K = seq_to_2d(global_K, size_2d)
            local_V = seq_to_2d(global_V, size_2d)
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory

        tgt2 = self.long_term_attn(curr_Q, global_K, global_V)[0]
        if self.drop_long_term:
            tgt2 = self.droppath(tgt2)

        tgt3 = self.short_term_attn(local_Q, local_K, local_V)[0]
        tgt3 = self.droppath(tgt3)

        tgt = tgt + tgt2 + tgt3

        # Feed-forward
        _tgt = self.norm3(tgt)

        tgt2 = self.linear2(
            self.activation(self.linear1(_tgt), size_2d))
        tgt2 = self.droppath(tgt2)

        tgt = tgt + tgt2

        return tgt, [[curr_K, curr_V], [global_K, global_V], [local_K, local_V]]


class MultiheadLocalAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0., max_dis=7, dilation=1, use_linear=True, enable_corr=True):
        super().__init__()
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.nhead = nhead
        self.T = ((d_model / nhead) ** 0.5)

        self.use_linear = use_linear
        if use_linear:
            self.Q = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
            self.K = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
            self.V = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)

        self.relative_emb = nn.Conv2d(
            d_model, nhead * self.window_size * self.window_size, kernel_size=1, bias=True, groups=nhead)
        self.V_bias = nn.Parameter(torch.zeros(
            [1, self.nhead, d_model // self.nhead, self.window_size * self.window_size, 1]))

        self.enable_corr = enable_corr

        if enable_corr:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.window_size,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=self.dilation)

        self.fc = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        n, c, h, w = v.size()

        if self.use_linear:
            q = self.Q(q)
            k = self.K(k)
            v = self.V(v)

        hidden_dim = c // self.nhead

        relative_emb = self.relative_emb(q)
        memory_mask = torch.ones((1, 1, h, w), device=v.device).float()

        # Scale
        q = q / self.T

        q = q.view(-1, hidden_dim, h, w)
        k = k.reshape(-1, hidden_dim, h, w).contiguous()
        unfolded_v = self.pad_and_unfold(v).view(
            n, self.nhead, hidden_dim, self.window_size * self.window_size, h * w) + self.V_bias

        relative_emb = relative_emb.view(
            n, self.nhead, self.window_size * self.window_size, h * w)
        unfolded_k_mask = self.pad_and_unfold(memory_mask).bool().view(1, 1,
                                                                       self.window_size * self.window_size, h * w).expand(n, self.nhead, -1, -1)

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(n, self.nhead, self.window_size *
                                                     self.window_size, h * w) + relative_emb
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.nhead, hidden_dim, self.window_size * self.window_size, h, w)
            qk = (q.unsqueeze(2) * unfolded_k).sum(dim=1).view(n, self.nhead, self.window_size *
                                                               self.window_size, h * w) + relative_emb

        qk[~unfolded_k_mask] = -1e+8 if qk.dtype == torch.float32 else -1e+4

        attn = torch.softmax(qk, dim=2)

        attn = self.dropout(attn)

        output = (attn.unsqueeze(2) * unfolded_v).sum(dim=3).permute(3,
                                                                     0, 1, 2).view(h * w, n, c)

        output = self.fc(output)

        return output, attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis * self.dilation
        x = F.pad(x,
                  (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                  mode='constant', value=0)
        x = F.unfold(x, kernel_size=(self.window_size, self.window_size),
                     stride=(1, 1), dilation=self.dilation)
        return x

class MultiheadLocalAttentionV2(nn.Module):

    def __init__(self, d_model, nhead, dropout=0., max_dis=7, dilation=1, use_linear=True, enable_corr=True):
        super().__init__()
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.nhead = nhead
        self.T = ((d_model / nhead) ** 0.5)

        self.use_linear = use_linear
        if use_linear:
            self.Q = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
            self.K = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
            self.V = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)

        self.relative_emb = nn.Conv2d(
            d_model, nhead * self.window_size * self.window_size, kernel_size=1, bias=True, groups=nhead)
        self.V_bias = nn.Parameter(torch.zeros(
            [1, self.nhead, d_model // self.nhead, self.window_size * self.window_size, 1]))

        self.enable_corr = enable_corr

        if enable_corr:
            from spatial_correlation_sampler import SpatialCorrelationSampler
            self.correlation_sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.window_size,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=self.dilation)

        self.fc = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.local_mask = None
        self.size_2d = None

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def forward(self, q, k, v):
        n, c, h, w = v.size()

        if self.use_linear:
            q = self.Q(q)
            k = self.K(k)
            v = self.V(v)

        hidden_dim = c // self.nhead

        relative_emb = self.relative_emb(q)
        memory_mask = torch.ones((1, 1, h, w), device=v.device).float()

        v = v.view(-1, self.nhead, hidden_dim, h * w)

        # Scale
        q = q / self.T

        q = q.view(-1, hidden_dim, h, w)
        k = k.reshape(-1, hidden_dim, h, w).contiguous()

        relative_emb = relative_emb.view(
            n, self.nhead, self.window_size * self.window_size, h * w)
        unfolded_k_mask = self.pad_and_unfold(memory_mask).bool().view(1, 1,
                                                                       self.window_size * self.window_size, h * w).expand(n, self.nhead, -1, -1)

        if self.enable_corr:
            qk = self.correlation_sampler(q, k).view(n, self.nhead, self.window_size *
                                                     self.window_size, h * w) + relative_emb
        else:
            unfolded_k = self.pad_and_unfold(k).view(
                n * self.nhead, hidden_dim, self.window_size * self.window_size, h, w)
            qk = (q.unsqueeze(2) * unfolded_k).sum(dim=1).view(n, self.nhead, self.window_size *
                                                               self.window_size, h * w) + relative_emb

        qk[~unfolded_k_mask] = -1e+8 if qk.dtype == torch.float32 else -1e+4

        attn = torch.softmax(qk, dim=2)

        attn = self.dropout(attn)

        agg_bias = torch.einsum('bhwn,hcw->nbhc', attn,
                                self.V_bias.squeeze(0).squeeze(-1)).reshape(h * w, n, c)

        global_attn = self.local2global(attn, h, w)

        agg_value = torch.einsum('bhnm,bhcm->nbhc', global_attn, v).reshape(h * w, n, c)

        output = agg_value + agg_bias

        output = self.fc(output)

        return output, attn

    def local2global(self, local_attn, height, width):
        batch_size = local_attn.size()[0]

        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.local_mask is not None and (height, width) == self.size_2d:
            local_mask = self.local_mask
        else:
            ky, kx = torch.meshgrid([torch.arange(0, pad_height, device=local_attn.device),
                                     torch.arange(0, pad_width, device=local_attn.device)])
            qy, qx = torch.meshgrid([torch.arange(0, height, device=local_attn.device),
                                     torch.arange(0, width, device=local_attn.device)])

            offset_y = qy.reshape(-1, 1) - ky.reshape(1, -1) + self.max_dis
            offset_x = qx.reshape(-1, 1) - kx.reshape(1, -1) + self.max_dis

            local_mask = (offset_y.abs() <= self.max_dis) & (offset_x.abs() <= self.max_dis)
            local_mask = local_mask.view(1, 1, height * width, pad_height, pad_width)
            self.local_mask = local_mask
            self.size_2d = (height, width)

        global_attn = torch.zeros((batch_size, self.nhead, height * width, pad_height, pad_width), device=local_attn.device)
        global_attn.masked_scatter_(local_mask, local_attn.transpose(-1, -2))
        global_attn = global_attn[:, :, :, self.max_dis:-self.max_dis, self.max_dis:-self.max_dis].reshape(batch_size, self.nhead, height * width, height * width)


        return global_attn

    def pad_and_unfold(self, x):
        pad_pixel = self.max_dis * self.dilation
        x = F.pad(x,
                  (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                  mode='constant', value=0)
        x = F.unfold(x, kernel_size=(self.window_size, self.window_size),
                     stride=(1, 1), dilation=self.dilation)
        return x

class MultiheadAttention(nn.Module):

    def __init__(self,
                 d_model,
                 nhead=8,
                 dropout=0.,
                 use_linear=True):
        super().__init__()
        self.d_model = d_model

        self.hidden_dim = d_model // nhead
        self.T = (d_model / nhead) ** 0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.nhead = nhead
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_heads = self.nhead
        hidden_dim = self.hidden_dim

        bs = Q.size()[1]

        # Linear projections
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        # Multi-head
        Q = Q.view(-1, bs, num_heads, hidden_dim).permute(1, 2, 0, 3)
        K = K.view(-1, bs, num_heads, hidden_dim).permute(1, 2, 3, 0)
        V = V.view(-1, bs, num_heads, hidden_dim).permute(1, 2, 0, 3)

        # Scale
        Q = Q / self.T

        # Multiplication
        QK = Q @ K

        # Activation
        attn = torch.softmax(QK, dim=-1)

        # Dropouts
        attn = self.dropout(attn)

        # Weighted sum
        outputs = (attn @ V).permute(2, 0, 1, 3)

        # Restore shape
        outputs = outputs.reshape(-1, bs, self.d_model)

        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
