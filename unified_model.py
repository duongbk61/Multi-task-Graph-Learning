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
        # Xét theo bảng: Call (Dim 0-6), Trans (Dim 7-13)
        # Dim 0: Tổng giá trị gửi đi (Call)
        # Dim 6: Tần suất nhận (Call)
        # Dim 11: Số dư (Trans)
        
        total_sent_call = features[:, 0]
        freq_recv_call = features[:, 6]
        balance_trans = features[:, 11]
        
        # Ponzi thường có tổng gửi đi lớn (trả lãi bùng nổ), và số dư rỗng hoặc bất thường
        score = (total_sent_call > 15).float() * 0.4 + \
                (balance_trans > 15).float() * 0.4 + \
                (freq_recv_call <= 0).float() * 0.2
        return score

    @staticmethod
    def compute_phish_score(features):
        # Nhóm đặc trưng Transaction (Dim 7-13)
        # Dim 7: Tổng giá trị gửi đi (Trans)
        # Dim 8: Tổng giá trị nhận về (Trans)
        # Dim 12: Tần suất khởi tạo/gửi đi (Trans)
        
        total_sent_trans = features[:, 7]
        total_recv_trans = features[:, 8]
        freq_sent_trans = features[:, 12]
        
        # Phishing thường là dạng tài khoản "Burner" (Tuổi thọ giao dịch siêu ngắn)
        # Tần suất rất thấp và Tổng giá trị giao dịch rất nhỏ do bị block hoặc tẩu tán nhanh
        score = (total_sent_trans < 100).float() * 0.35 + \
                (total_recv_trans < 100).float() * 0.35 + \
                (freq_sent_trans < 10).float() * 0.30
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
