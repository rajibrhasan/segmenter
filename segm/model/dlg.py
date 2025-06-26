import torch
import torch.nn as nn

class DenseLanguageGuidanceModule(nn.Module):
    def __init__(self, v_feature_dim, l_feature_dim, feature_dim, final_dim, nl = 77):
        super().__init__()
        self.feature_dim = feature_dim

        # Linear layers for key/value projections
        self.linear_vk = nn.Linear(v_feature_dim, feature_dim)
        self.linear_vv = nn.Linear(v_feature_dim, feature_dim)
        self.linear_lk = nn.Linear(l_feature_dim, feature_dim)
        self.linear_lv = nn.Linear(l_feature_dim, feature_dim)

        self.linear_m = nn.Linear(nl, 768)

    def forward(self, fv, fl):
        # fv: (batch_size, h*w, c)
        # fl: (batch_size, nL, c)

        # 1. Project to Key-Value Pairs
        fk_v = self.linear_vk(fv)
        fv_v = self.linear_vv(fv)
        fk_l = self.linear_lk(fl)
        fv_l = self.linear_lv(fl)

        # 2. Attention Matrix Calculation (A = (1/sqrt(c)) FK_L @ FK_V^T)
        # Assuming A should be nL x HW as stated: A ∈ R^(nL×h×w)
        # So, language queries (FK_L) attend to vision keys (FK_V)
        # fk_l: (B, nL, C)
        # fk_v: (B, HW, C)
        # fk_v.transpose(-1, -2): (B, C, HW)
        a_raw = torch.matmul(fk_l, fk_v.transpose(-1, -2)) / (self.feature_dim**0.5) # (B, nL, HW)

        # 3. Cross-Attention on Vision and Language Features
        # Language-attended vision features (language queries attend to vision values)
        # softmax applied across the HW dimension for each nL query
        fa_v = torch.softmax(a_raw, dim=-1) @ fv_v # (B, nL, HW) @ (B, HW, C) -> (B, nL, C)

        # Vision-attended language features (vision queries attend to language values)
        # Need to transpose A_raw to make vision the query and language the key
        # a_raw.transpose(-1, -2): (B, HW, nL)
        fa_l = torch.softmax(a_raw.transpose(-1, -2), dim=-1) @ fv_l # (B, HW, nL) @ (B, nL, C) -> (B, HW, C)

        # 4. Combine Attended Feature Maps
        # FM = FA_V . (FA_L)^T
        # fa_v: (B, nL, C)
        # fa_l: (B, HW, C)
        # fa_l.transpose(-1, -2): (B, C, HW)
        fm = torch.matmul(fa_v, fa_l.transpose(-1, -2)) # (B, nL, C) @ (B, C, HW) -> (B, nL, HW)
        fm = fm.transpose(1, 2)
        fm = self.linear_m(fm)

        return fm
    