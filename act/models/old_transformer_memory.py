import torch
import torch.nn as nn


class OLDTransformerMemory(nn.Module):
    def __init__(self, embed_dim, num_heads, memory_size):
        super().__init__()

        self.static_attention = nn.MultiheadAttention(embed_dim, num_heads=1)
        self.memory_attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads)

        self.feedback_layer_attention = nn.MultiheadAttention(embed_dim, num_heads=1)
        self.feedback_global_attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads)
        self.dim_reduce = nn.Linear(2 * embed_dim, embed_dim)
        self.gate_linear = nn.Linear(embed_dim, embed_dim)

        # Memory will be initialized on the first call
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.memory = None

    def forward(self, x):
        # If memory is not initialized, do it based on batch size
        if self.memory is None:
            batch_size = x.size(1)
            self.memory = nn.Parameter(torch.randn(self.memory_size, batch_size, self.embed_dim)).cuda()

        # print("DEBUG - X SHAPE:", x.shape)
        # print("DEBUG - MEMORY SHAPE:", self.memory.shape)
        # Static attention
        static_out, _ = self.static_attention(x, x, x)

        # Memory attention
        memory_out, _ = self.memory_attention(x, self.memory, self.memory)

        # Combining static and memory outputs
        combined_input = torch.cat([static_out, memory_out], dim=2)
        combined_input = self.dim_reduce(combined_input)
        cross_out, _ = self.cross_attention(x, combined_input, combined_input)

        return cross_out

    def update(self, transformer_out):
        # Reduce to top few layers
        top_layers_out = transformer_out[-3:]  # consider last 3 layers

        # Pooling over layers to reduce sequence length
        pooled_out = torch.max(top_layers_out, dim=2).values  # max pooling over sequence length

        # Layer-level attention
        layer_attentions = []
        for layer_out in pooled_out:
            # print("DEBUG LAYER OUT SHAPE:", layer_out.unsqueeze(0).shape)
            # print("DEBUG MEMORY SHAPE:", self.memory.shape)
            attention_out, _ = self.feedback_layer_attention(layer_out.unsqueeze(0), self.memory, self.memory)
            layer_attentions.append(attention_out)
        layer_attentions = torch.stack(layer_attentions, dim=0)

        # Global attention over layer attentions
        memory_update, _ = self.feedback_global_attention(torch.mean(layer_attentions, dim=0), self.memory, self.memory)

        # Gating mechanism
        gate = torch.sigmoid(self.gate_linear(memory_update))
        self.memory.data = gate * memory_update + (1 - gate) * self.memory.data

    def reset(self):
        if self.memory is not None:
            del self.memory
        self.memory = None
