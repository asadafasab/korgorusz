from korgorusz.layers import LayerNorm, Linear, ReLU, Dropout, Softmax


class MultiHeadAttention:
    def __init__(self, heads, d_model, d_k, d_v, dropout_rate=0.1):
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.embed_size_pow = d_k ** 0.5

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.softmax = Softmax()
        self.norm = LayerNorm(d_model)

        self.values_fc = Linear(d_model, heads * d_k, bias=False)
        self.keys_fc = Linear(d_model, heads * d_k, bias=False)
        self.queries_fc = Linear(d_model, heads * d_v, bias=False)
        self.out_fc = Linear(heads * d_v, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        derivatives = []
        d_k, d_v, heads = self.d_k, self.d_v, self.heads
        sz_b, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = q

        q, d = self.queries_fc.forward(q)
        derivatives.append(d)
        q = q.reshape(sz_b, len_q, heads, d_k)

        k, d = self.keys_fc.forward(k)
        derivatives.append(d)
        k = k.reshape(sz_b, len_k, heads, d_k)

        v, d = self.values_fc.forward(v)
        derivatives.append(d)
        v = v.reshape(sz_b, len_v, heads, d_v)

        q, k, v = q.transpose(0, 2, 1), k.transpose(0, 2, 1), v.transpose(0, 2, 1)

        # attention
        attention = (q / self.embed_size_pow) @ k.transpose(0, 1, 3, 2)
        attention = attention @ v

        if mask is not None:
            mask = mask.unsqueeze(1)
            attention[mask == 0] = -1e9

        attention, d = self.softmax.forward(attention, dim=-1)
        derivatives.append(d)

        attention, d = self.dropout1.forward(attention)
        derivatives.append(d)

        q = q.transpose(0, 2, 1).reshape(sz_b, len_q, -1)
        q, d = self.dropout2.forward(self.out_fc.forward(q))
        derivatives.append(d)
        q += residual

        q, d = self.norm.forward(q)
        derivatives.append(d)

        return q, attention, derivatives
