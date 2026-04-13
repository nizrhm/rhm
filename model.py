import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TRMBlock(nn.Module):
    """Upgraded to Multi-Head Attention for long sequence (ARC) efficiency."""
    def __init__(self, d_model, d_ff, seq_len, dropout=0.1):
        super().__init__()
        
        # 1. Spatial Mixing via Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=8, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 2. Deep Reasoning via SwiGLU
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + self.dropout(attn_out)
        
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class SelfEvolvingTRM(nn.Module):
    def __init__(self, vocab_size, d_model=256, d_ff=1024, n_layers=2, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList([TRMBlock(d_model, d_ff, max_seq_len) for _ in range(n_layers)])
        
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.unembed.weight = self.embedding.weight

    def _process_stream(self, combined_state):
        for layer in self.layers:
            combined_state = layer(combined_state)
        return combined_state

    def forward(self, x_ids, past_z=None, max_reasoning_steps=4, max_refinement_steps=8, early_exit_threshold=1e-4):
        batch_size, seq_len = x_ids.size()
        
        x_embedded = self.embedding(x_ids) + self.pos_encoding[:, :seq_len, :]
        x = x_embedded
        
        # RAG Memory Injection
        if past_z is not None:
            z = past_z.clone()
            current_reasoning_steps = 0 
        else:
            z = torch.randn_like(x) * 0.02 
            current_reasoning_steps = max_reasoning_steps
            
        y = torch.randn_like(x) * 0.02 
        trajectory_y = [] 
        
        # Phase 1: Reasoning
        for step in range(current_reasoning_steps):
            combined = x + y + z 
            z_new = self._process_stream(combined)
            z = z_new 
            
        # Phase 2: Refinement
        for step in range(max_refinement_steps):
            combined = x + y + z
            y_new = self._process_stream(combined)
            
            diff = torch.norm(y_new - y, p=2, dim=-1).mean()
            y = y_new
            trajectory_y.append(y.clone())
            
            if not self.training and diff < early_exit_threshold:
                break 
                
        logits_trajectory = [self.unembed(y_state) for y_state in trajectory_y]
        
        return logits_trajectory[-1], logits_trajectory, x_embedded, z