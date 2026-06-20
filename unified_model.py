from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import MessagePassing
from attention_conv import my_conv
# from model import TripletLoss

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''

    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda:
                y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)
        return loss


class ExpertRules:
    # 14-dim feature layout, verified empirically from the data
    # (total = count x avg ; balance = total_inv - total_ret, corr 1.0).
    # See feature_columns_definition.csv. Layout: [call: 0-6][trans: 7-13]
    #   idx  0 Call  Balance            (signed)
    #   idx  1 Call  Total investment
    #   idx  2 Call  Total return
    #   idx  3 Call  Count initiation   (int)
    #   idx  4 Call  Count reception    (int)
    #   idx  5 Call  Avg investment
    #   idx  6 Call  Avg return         (signed; -1 when count=0)
    #   idx  7 Trans Balance            (signed)
    #   idx  8 Trans Total investment
    #   idx  9 Trans Total return
    #   idx 10 Trans Count initiation   (int)
    #   idx 11 Trans Count reception    (int)
    #   idx 12 Trans Avg investment
    #   idx 13 Trans Avg return         (signed; -1 when count=0)

    @staticmethod
    def compute_ponzi_score(features):
        call_balance = features[:, 0]
        call_count_recv = features[:, 4]
        trans_balance = features[:, 7]
        trans_count_recv = features[:, 11]

        c1_positive_trans_balance = trans_balance > 0.0
        c2_ponzi_pattern = (call_count_recv <= 3.5) | (trans_count_recv > 269.0)
        branch_1 = c1_positive_trans_balance & c2_ponzi_pattern

        branch_2 = (trans_balance <= 0.0) & (call_balance > 10.0)

        score = (branch_1 | branch_2).float()
        return score

    @staticmethod
    def compute_phish_score(features):
        trans_total_inv = features[:, 8]
        trans_balance = features[:, 7]
        call_balance = features[:, 0]

        c1_low_investment = trans_total_inv <= 102.04
        c2_trans_balance = trans_balance > -10.0
        c3_call_balance = call_balance > -12.75
        branch_1 = c1_low_investment & c2_trans_balance & c3_call_balance

        call_total_ret = features[:, 2]
        call_avg_ret = features[:, 6]
        c_whale = trans_total_inv > 150.0
        c_no_call = (call_balance == 0.0) & (call_total_ret == 0.0) & (call_avg_ret == 0.0)
        branch_2 = c_whale & c_no_call

        score = (branch_1 | branch_2).float()
        return score


class CrossPathAttention(nn.Module):
    """
    Mutual attention between the generative path (h_gen) and the contrastive path (h_cont).

    Each path attends to the other via a gated residual:
      - h_gen'  = h_gen  + sigmoid(q_gen  · k_cont / sqrt(d)) * v_cont
      - h_cont' = h_cont + sigmoid(q_cont · k_gen  / sqrt(d)) * v_gen

    h_gen  learns what it missed from the contrastive path.
    h_cont learns what it missed from the generative path.
    The residual connection preserves each path's own signal.
    """
    def __init__(self, hidden, dropout=0.3):
        super().__init__()
        self.scale = hidden ** 0.5
        # h_gen attending to h_cont
        self.q_gen  = nn.Linear(hidden, hidden)
        self.k_cont = nn.Linear(hidden, hidden)
        self.v_cont = nn.Linear(hidden, hidden)
        # h_cont attending to h_gen
        self.q_cont = nn.Linear(hidden, hidden)
        self.k_gen  = nn.Linear(hidden, hidden)
        self.v_gen  = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_gen, h_cont):
        # h_gen queries h_cont
        alpha_gc = torch.sigmoid(
            (self.q_gen(h_gen) * self.k_cont(h_cont)).sum(dim=-1, keepdim=True) / self.scale
        )
        h_gen_prime = h_gen + alpha_gc * self.dropout(self.v_cont(h_cont))

        # h_cont queries h_gen
        alpha_cg = torch.sigmoid(
            (self.q_cont(h_cont) * self.k_gen(h_gen)).sum(dim=-1, keepdim=True) / self.scale
        )
        h_cont_prime = h_cont + alpha_cg * self.dropout(self.v_gen(h_gen))

        return h_gen_prime, h_cont_prime

    def reset_parameters(self):
        for layer in [self.q_gen, self.k_cont, self.v_cont,
                      self.q_cont, self.k_gen, self.v_gen]:
            layer.reset_parameters()


class TaskGate(nn.Module):
    """
    Task-conditioned gate that blends h_gen' and h_cont' using raw node features.
    A separate TaskGate per node type means the Ponzi task (CA) and Phishing task (EOA)
    learn different blending strategies from the same feature space.

    alpha = sigmoid(MLP(raw_x))
    h_fused = alpha * h_gen' + (1 - alpha) * h_cont'
    """
    def __init__(self, hidden, dropout=0.3):
        super().__init__()
        self.fc1 = Linear(-1, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, raw_x, h_gen_prime, h_cont_prime):
        alpha = torch.sigmoid(self.fc2(self.dropout(F.relu(self.fc1(raw_x)))))  # [N, 1]
        return alpha * h_gen_prime + (1 - alpha) * h_cont_prime

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class UnifiedHMSL(MessagePassing):
    def __init__(self, hidden, out_channels, data, concat, expert_mode='feature'):
        super().__init__(aggr='sum')
        self.expert_mode = expert_mode
        self.concat_num = concat
        self.loss_co = TripletLoss(margin=0.3)

        self.lin_dict = torch.nn.ModuleDict()
        self.lin_dict_mean = torch.nn.ModuleDict()
        self.lin_orig = torch.nn.ModuleDict()   # contrastive path encoder
        self.cross_attn = torch.nn.ModuleDict() # Option 1: cross-path mutual attention
        self.task_gate = torch.nn.ModuleDict()  # Option 2: task-conditioned gate

        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden)
            self.lin_dict_mean[node_type] = Linear(hidden, hidden)
            self.lin_orig[node_type] = Linear(-1, hidden)
            self.cross_attn[node_type] = CrossPathAttention(hidden)
            self.task_gate[node_type] = TaskGate(hidden)

        # Multi-task Heads
        head_in = hidden + 1 if expert_mode == 'feature' else hidden
        self.head_ponzi = Linear(head_in, out_channels)
        self.head_phish = Linear(head_in, out_channels)

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()

        for node_type in data.node_types:
            self.k_lin[node_type] = Linear(hidden, hidden)
            self.q_lin[node_type] = Linear(hidden, hidden)
            self.v_lin[node_type] = Linear(hidden, hidden)

        self.conv  = my_conv(hidden, hidden, metadata=data.metadata())
        self.conv1 = my_conv(hidden, hidden, metadata=data.metadata())

    def reset_parameters(self):
        reset(self.lin_dict)
        reset(self.lin_dict_mean)
        reset(self.lin_orig)
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.conv)
        reset(self.conv1)
        self.head_ponzi.reset_parameters()
        self.head_phish.reset_parameters()
        for node_type in self.cross_attn:
            self.cross_attn[node_type].reset_parameters()
        for node_type in self.task_gate:
            self.task_gate[node_type].reset_parameters()

    def forward(self, x_dict, edge_index, raw_x_dict=None):
        CA_hidden_ls  = []
        EOA_hidden_ls = []
        ca_out_ls     = []
        eoa_out_ls    = []
        out_dict      = {}

        # --- Build multi-view embeddings from CVAE-augmented features ---
        for k in range(self.concat_num):
            x_dict_ = {
                node_type: F.tanh(self.lin_dict[node_type](x[k]))
                for node_type, x in x_dict.items()
            }
            CA_hidden_ls.append(x_dict_['CA'])
            EOA_hidden_ls.append(x_dict_['EOA'])
            CA_h  = F.tanh(self.lin_dict_mean['CA'] (torch.mean(CA_hidden_ls[k],  dim=0)))
            EOA_h = F.tanh(self.lin_dict_mean['EOA'](torch.mean(EOA_hidden_ls[k], dim=0)))
            ca_out_ls.append(CA_h)
            eoa_out_ls.append(EOA_h)

        # --- Generative path: self-attention over CVAE stochastic views ---
        h_gen_ca  = self.attention(ca_out_ls,  'CA')
        h_gen_eoa = self.attention(eoa_out_ls, 'EOA')

        if raw_x_dict is not None:
            # --- Contrastive path: linear transform of original raw features ---
            h_cont_ca  = F.tanh(self.lin_orig['CA'] (raw_x_dict['CA']))
            h_cont_eoa = F.tanh(self.lin_orig['EOA'](raw_x_dict['EOA']))

            # --- Option 1: Cross-path mutual attention ---
            h_gen_ca_prime,  h_cont_ca_prime  = self.cross_attn['CA'] (h_gen_ca,  h_cont_ca)
            h_gen_eoa_prime, h_cont_eoa_prime = self.cross_attn['EOA'](h_gen_eoa, h_cont_eoa)

            # --- Option 2: Task-conditioned gate ---
            out_dict['CA']  = self.task_gate['CA'] (raw_x_dict['CA'],  h_gen_ca_prime,  h_cont_ca_prime)
            out_dict['EOA'] = self.task_gate['EOA'](raw_x_dict['EOA'], h_gen_eoa_prime, h_cont_eoa_prime)
        else:
            out_dict['CA']  = h_gen_ca
            out_dict['EOA'] = h_gen_eoa

        # --- Graph convolution ---
        out_dict = self.conv(out_dict,  edge_index)
        out_dict = self.conv1(out_dict, edge_index)

        # --- Expert rules ---
        if raw_x_dict is not None and self.expert_mode != 'none':
            expert_ponzi = ExpertRules.compute_ponzi_score(raw_x_dict['CA']).unsqueeze(1).to(out_dict['CA'].device)
            expert_phish = ExpertRules.compute_phish_score(raw_x_dict['EOA']).unsqueeze(1).to(out_dict['EOA'].device)
        else:
            expert_ponzi = torch.zeros((out_dict['CA'].shape[0],  1), device=out_dict['CA'].device)
            expert_phish = torch.zeros((out_dict['EOA'].shape[0], 1), device=out_dict['EOA'].device)

        # --- Classification heads ---
        if self.expert_mode == 'feature':
            ca_final  = torch.cat([out_dict['CA'],  expert_ponzi], dim=-1)
            eoa_final = torch.cat([out_dict['EOA'], expert_phish], dim=-1)
        else:
            ca_final  = out_dict['CA']
            eoa_final = out_dict['EOA']

        out_ponzi = self.head_ponzi(ca_final)
        out_phish = self.head_phish(eoa_final)
        loss_co   = self.contrast_module(CA_hidden_ls, EOA_hidden_ls)

        return out_ponzi, out_phish, loss_co, expert_ponzi, expert_phish

    def contrast_module(self, CA_hidden_ls, EOA_hidden_ls):
        anchors, pos, neg = [], [], []
        for z in CA_hidden_ls:
            anchors.append(z[1])
            pos.append(z[2])
            neg.append(z[0])
        loss_ca = self.loss_co(
            self.attention(anchors, 'CA'),
            self.attention(pos,     'CA'),
            self.attention(neg,     'CA'),
        )

        anchors, pos, neg = [], [], []
        for z in EOA_hidden_ls:
            anchors.append(z[1])
            pos.append(z[2])
            neg.append(z[0])
        loss_eoa = self.loss_co(
            self.attention(anchors, 'EOA'),
            self.attention(pos,     'EOA'),
            self.attention(neg,     'EOA'),
        )
        return loss_ca + loss_eoa

    def attention(self, input, node_type):
        input = torch.stack(input, dim=0).permute(1, 0, 2)
        q = self.q_lin[node_type](input)
        k = self.k_lin[node_type](input)
        v = self.v_lin[node_type](input)
        att = F.softmax(torch.matmul(k, q.transpose(1, 2)), dim=-1)
        return torch.mean(att @ v, dim=1)
