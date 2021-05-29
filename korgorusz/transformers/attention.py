from korgorusz.layers import LayerNorm, Linear, ReLU, Dropout


class ScaledAttention:
    def __init__(self, temperature, dropout_rate=0.1):
        ...


##########################################################
# TODO ___________________________________________________
# TODO rename variables to something that makes more sense
# TODO ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##########################################################


class MultiHeadAttention:
    def __init__(self, heads, d_model, d_k, d_v, dropout_rate=0.1):
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.embed_size_pow = d_k ** 0.5

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.norm = LayerNorm(d_model)

        self.values = Linear(d_model, heads * d_k, bias=False)
        self.keys = Linear(d_model, heads * d_k, bias=False)
        self.queries = Linear(d_model, heads * d_v, bias=False)
        self.fc_out = Linear(heads * d_v, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        ...
