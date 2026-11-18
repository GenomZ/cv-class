"""Reference entries for functions first used in P8 (vision transformers).

Same format as `_day0.py`. Entries whose key already exists in an earlier file are ignored (the earliest
wording wins), so P8 reuses the earlier entries for nn.Parameter, Tensor.expand, torchvision.models.vit_b_16, ...
and adds only the transformer layers no earlier notebook explains.
"""
from . import E

ENTRIES = [
    E("nn.MultiheadAttention", "nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)",
      """An attention layer. Called as attention(query, key, value) with three tensors of shape (batch, tokens,
      embed_dim); for self-attention, the kind a vision transformer uses, all three are the same tokens. Inside,
      three linear maps make a query, a key and a value from every token; the scores query . key (divided by
      the square root of the numbers per head) go through a softmax, and every token becomes the weighted
      average of the values. It returns two things: the new tokens (same shape as the input) and the weights
      (batch, tokens, tokens), averaged over the heads; each row of the weights adds up to 1. num_heads splits
      the embed_dim numbers into that many independent attentions side by side. Parameters: 4 x embed_dim x
      embed_dim weights + 4 x embed_dim biases.""",
      """
      >>> attention = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
      >>> tokens = torch.randn(1, 5, 8)                    # 1 picture, 5 tokens of 8 numbers
      >>> new_tokens, weights = attention(tokens, tokens, tokens)
      >>> new_tokens.shape, weights.shape
      (torch.Size([1, 5, 8]), torch.Size([1, 5, 5]))
      >>> torch.allclose(weights.sum(dim=2), torch.ones(1, 5))     # every row of weights adds up to 1
      True
      >>> sum(p.numel() for p in attention.parameters())  # 4 x 8 x 8 + 4 x 8
      288
      """,
      note="""Without batch_first=True the layer expects (tokens, batch, embed_dim), the old order of language
      models. embed_dim must be divisible by num_heads.""",
      see=["nn.TransformerEncoderLayer", "Tensor.softmax"]),

    E("nn.LayerNorm", "nn.LayerNorm(normalized_shape)",
      """Layer normalisation: rescales the numbers of every token (the last axis, normalized_shape numbers) to an
      average of 0 and a spread of 1, then multiplies by a learned scale and adds a learned shift (2 x
      normalized_shape parameters). Unlike batch normalisation it looks at one token at a time, never at the
      other pictures of the batch, so it works the same in training and in prediction. Transformers use it
      before every attention and every MLP.""",
      """
      >>> norm = nn.LayerNorm(4)
      >>> tokens = torch.tensor([[1.0, 2.0, 3.0, 4.0],
      ...                        [10.0, 10.0, 10.0, 30.0]])
      >>> out = norm(tokens)
      >>> out.mean(dim=1).abs().max() < 1e-6               # every row now averages 0
      tensor(True)
      >>> out.round(decimals=2)
      tensor([[-1.3400, -0.4500,  0.4500,  1.3400],
              [-0.5800, -0.5800, -0.5800,  1.7300]], grad_fn=<RoundBackward1>)
      >>> sum(p.numel() for p in norm.parameters())       # a scale and a shift per number
      8
      """,
      see=["nn.BatchNorm1d", "nn.TransformerEncoderLayer"]),

    E("nn.TransformerEncoderLayer",
      'nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.0, activation="gelu", '
      'batch_first=True, norm_first=True)',
      """One transformer block, ready made. With norm_first=True (as in the vision transformer) it computes
      x = x + attention(layer_norm(x)), then x = x + mlp(layer_norm(x)): a layer norm, multi-head
      self-attention (nhead heads) and a skip connection, then a layer norm, a small MLP (Linear d_model ->
      dim_feedforward, the activation, Linear back to d_model) and a second skip connection. Input and output
      have the same shape (batch, tokens, d_model), so blocks can be stacked. Its parts are .self_attn,
      .linear1, .linear2, .norm1, .norm2.""",
      """
      >>> block = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0,
      ...                                    activation="gelu", batch_first=True, norm_first=True)
      >>> tokens = torch.randn(3, 5, 8)                    # 3 pictures, 5 tokens of 8 numbers
      >>> block(tokens).shape                              # the same shape comes out
      torch.Size([3, 5, 8])
      >>> sum(p.numel() for p in block.self_attn.parameters())     # attention: 4 x 8 x 8 + 4 x 8
      288
      >>> sum(p.numel() for p in block.parameters())       # + MLP (8 x 16 + 16 + 16 x 8 + 8) + 2 layer norms (2 x 16)
      600
      """,
      note="""The defaults are dropout=0.1, activation="relu", batch_first=False and norm_first=False (the
      original 2017 transformer for language): write all four as above for a ViT. d_model must be divisible by
      nhead.""",
      see=["nn.MultiheadAttention", "nn.LayerNorm"]),
]
