import torch
import torch.nn as nn


class TransformerMemory(nn.Module):
    def __init__(self, embed_dim, num_heads, memory_size):
        super().__init__()

        self.memory_pos_embedding = nn.Embedding(memory_size, embed_dim)
        self.feedback_attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.gate_linear = nn.Linear(embed_dim, embed_dim)

        # Memory will be initialized on the first call
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.memory = None

    def get(self, x):
        if self.memory is None:
            batch_size = x.size(1)
            self.memory = nn.Parameter(torch.randn(self.memory_size, batch_size, self.embed_dim)).cuda()
        return self.memory, self.memory_pos_embedding.weight

    def update(self, transformer_out):
        last_layer = transformer_out[-1].transpose(1, 0)
        # Attention
        memory_update, _ = self.feedback_attention(last_layer, self.memory, self.memory)

        # Gating mechanism
        gate = torch.sigmoid(self.gate_linear(memory_update))

        self.memory.data = gate * memory_update + (1 - gate) * self.memory.data

    def reset(self):
        if self.memory is not None:
            del self.memory
        self.memory = None
