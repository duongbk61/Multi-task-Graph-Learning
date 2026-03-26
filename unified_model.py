from typing import Optional, List
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import MessagePassing
from attention_conv import my_conv
from model import TripletLoss

class ExpertRules:
    @staticmethod
    def compute_ponzi_score(features):
        # features shape: [N, 14]
        # Xét theo bảng Data Features:
        # Dim 0: Call_Total_Sent
        # Dim 4: Call_Balance
        # Dim 7: Trans_Total_Sent
        # Dim 11: Trans_Balance
        
        call_total_sent = features[:, 0]
        call_balance = features[:, 4]
        trans_total_sent = features[:, 7]
        trans_balance = features[:, 11]
        
        # Áp dụng bộ luật bóc tách trực tiếp (Decision Tree + Error Analysis)
        # Nhánh 1: CÓ chuyển tiền gốc rải rác và thỏa mãn balance (luật cũ)
        c1_active_trans = trans_total_sent > 0.0
        c2_ponzi_pattern = (call_balance <= 3.5) | (trans_balance > 269.0)
        branch_1 = c1_active_trans & c2_ponzi_pattern
        
        # Nhánh 2: KHÔNG có giao dịch gửi thông thường, nhưng gọi contract RẤT LỚN (để hút/chuyển tiền)
        # Khức phục 8 case FNs đều có Trans_Total <= 0 & Call_Total cực cao (~45 mean).
        branch_2 = (trans_total_sent <= 0.0) & (call_total_sent > 10.0)
        
        score = (branch_1 | branch_2).float()
        return score

    @staticmethod
    def compute_phish_score(features):
        trans_total_recv = features[:, 8]
        trans_total_sent = features[:, 7]
        call_total_sent = features[:, 0]
        
        # Nhánh 1: Luật Burner Account (Phishing nhận ít rồi chuồn)
        c1_low_deposit = trans_total_recv <= 102.04
        c2_active_sent = trans_total_sent > -10.0
        c3_active_call = call_total_sent > -12.75
        branch_1 = c1_low_deposit & c2_active_sent & c3_active_call
        
        # Nhánh 2: Nhóm "Whale Phishing" (cá mập) - chuyên lừa đảo nhận rất nhiều tiền 
        # Khắc phục FNs: trans_total_recv rất cao (>150), nhưng không hề có các lệnh call (dim 0, 2, 6 == 0)
        call_balance = features[:, 2]
        feature_6 = features[:, 6]
        c_whale = trans_total_recv > 150.0
        c_no_call = (call_total_sent == 0.0) & (call_balance == 0.0) & (feature_6 == 0.0)
        branch_2 = c_whale & c_no_call
        
        score = (branch_1 | branch_2).float()
        return score


class UnifiedHMSL(MessagePassing):
    def __init__(self, hidden, out_channels, data, concat, expert_mode='feature'):
        super().__init__(aggr='sum')
        self.expert_mode = expert_mode
        self.concat_num = concat
        self.loss_co = TripletLoss(margin=0.3)

        self.lin_dict = torch.nn.ModuleDict()
        self.lin_dict_mean = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden)
            self.lin_dict_mean[node_type] = Linear(hidden, hidden)
        
        # Multi-task Heads
        head_in = hidden + 1 if expert_mode == 'feature' else hidden
        self.head_ponzi = Linear(head_in, out_channels) # For CA nodes
        self.head_phish = Linear(head_in, out_channels) # For EOA nodes

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()

        for node_type in data.node_types:
            self.k_lin[node_type] = Linear(hidden, hidden)
            self.q_lin[node_type] = Linear(hidden, hidden)
            self.v_lin[node_type] = Linear(hidden, hidden)

        self.conv = my_conv(hidden, hidden, metadata=data.metadata())
        self.conv1 = my_conv(hidden, hidden, metadata=data.metadata())

    def reset_parameters(self):
        reset(self.lin_dict)
        reset(self.lin_dict_mean)
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.conv)
        reset(self.conv1)
        self.head_ponzi.reset_parameters()
        self.head_phish.reset_parameters()

    def forward(self, x_dict, edge_index, raw_x_dict=None):
        CA_hidden_ls = []
        EOA_hidden_ls = []

        ca_out_ls = []
        eoa_out_ls = []
        out_dict = {}
        for k in range(self.concat_num):
            x_dict_ = {
                node_type: F.tanh(self.lin_dict[node_type](x[k]))
                for node_type, x in x_dict.items()
            }
            CA_hidden_ls.append(x_dict_['CA'])
            EOA_hidden_ls.append(x_dict_['EOA'])
            CA_h = torch.mean(CA_hidden_ls[k], dim=0)
            CA_h = F.tanh(self.lin_dict_mean['CA'](CA_h))
            EOA_h = torch.mean(EOA_hidden_ls[k], dim=0)
            EOA_h = F.tanh(self.lin_dict_mean['EOA'](EOA_h))
            ca_out_ls.append(CA_h)
            eoa_out_ls.append(EOA_h)

        out_dict['CA'] = ca_out_ls
        out_dict['EOA'] = eoa_out_ls

        for node_type, x in out_dict.items():
            out_dict[node_type] = self.attention(x, node_type)

        out_dict = self.conv(out_dict, edge_index)
        out_dict = self.conv1(out_dict, edge_index)

        expert_ponzi = None
        expert_phish = None
        if raw_x_dict is not None and self.expert_mode != 'none':
            expert_ponzi = ExpertRules.compute_ponzi_score(raw_x_dict['CA']).unsqueeze(1).to(out_dict['CA'].device)
            expert_phish = ExpertRules.compute_phish_score(raw_x_dict['EOA']).unsqueeze(1).to(out_dict['EOA'].device)
        else:
            expert_ponzi = torch.zeros((out_dict['CA'].shape[0], 1), device=out_dict['CA'].device)
            expert_phish = torch.zeros((out_dict['EOA'].shape[0], 1), device=out_dict['EOA'].device)

        # Multi-task Outputs
        if self.expert_mode == 'feature':
            ca_final = torch.cat([out_dict['CA'], expert_ponzi], dim=-1)
            eoa_final = torch.cat([out_dict['EOA'], expert_phish], dim=-1)
        else:
            ca_final = out_dict['CA']
            eoa_final = out_dict['EOA']

        out_ponzi = self.head_ponzi(ca_final)
        out_phish = self.head_phish(eoa_final)
        
        loss_co = self.contrast_module(CA_hidden_ls, EOA_hidden_ls)

        return out_ponzi, out_phish, loss_co, expert_ponzi, expert_phish

    def contrast_module(self, CA_hidden_ls, EOA_hidden_ls):
        anchors = []
        pos = []
        neg = []
        for i, z in enumerate(CA_hidden_ls):
            anchors.append(z[1])
            pos.append(z[2])
            neg.append(z[0])

        out_anchors = self.attention(anchors, 'CA')
        out_pos = self.attention(pos, 'CA')
        out_neg = self.attention(neg, 'CA')
        loss_ca = self.loss_co(out_anchors, out_pos, out_neg)

        anchors = []
        pos = []
        neg = []
        for i, z in enumerate(EOA_hidden_ls):
            anchors.append(z[1])
            pos.append(z[2])
            neg.append(z[0])

        out_anchors = self.attention(anchors, 'EOA')
        out_pos = self.attention(pos, 'EOA')
        out_neg = self.attention(neg, 'EOA')
        loss_eoa = self.loss_co(out_anchors, out_pos, out_neg)
        loss = loss_ca + loss_eoa
        return loss

    def attention(self, input, node_type):
        input = torch.stack(input, dim=0).permute(1, 0, 2)
        q = self.q_lin[node_type](input)
        k = self.k_lin[node_type](input)
        v = self.v_lin[node_type](input)
        att = torch.matmul(k, q.transpose(1, 2))
        att = torch.nn.functional.softmax(att, dim=-1)
        v = att @ v
        v = torch.mean(v, dim=1)
        return v
