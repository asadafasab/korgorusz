"""
Implementation of attention from Transformers
"""

from korgorusz.layers import LayerNorm, Linear, Dropout, Softmax


class MultiHeadAttention:
    """
    Runs through an attention mechanism several times.
    The independent attention outputs are then
    concatenated and linearly transformed into the expected dimension.
    """

    def __init__(self, heads, d_model, d_keys, d_values, dropout_rate=0.1):
        self.heads = heads
        self.d_keys = d_keys
        self.d_values = d_values
        self.embed_size_pow = d_keys ** 0.5

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.softmax = Softmax()
        self.norm = LayerNorm(d_model)

        self.values_fc = Linear(d_model, heads * d_keys, bias=False)
        self.keys_fc = Linear(d_model, heads * d_keys, bias=False)
        self.queries_fc = Linear(d_model, heads * d_values, bias=False)
        self.out_fc = Linear(heads * d_values, d_model, bias=False)

    def forward(self, query, keys, values, mask=None):
        """
        Calculates the query,keys,values...
        """
        derivatives = []
        d_k, d_v, heads = self.d_keys, self.d_values, self.heads
        sz_b, len_q, len_k, len_v = (
            query.shape[0],
            query.shape[1],
            keys.shape[1],
            values.shape[1],
        )

        residual = query

        query, deriv = self.queries_fc.forward(query)
        derivatives.append(deriv)
        query = query.reshape(sz_b, len_q, heads, d_k)

        keys, deriv = self.keys_fc.forward(keys)
        derivatives.append(deriv)
        keys = keys.reshape(sz_b, len_k, heads, d_k)

        values, deriv = self.values_fc.forward(values)
        derivatives.append(deriv)
        values = values.reshape(sz_b, len_v, heads, d_v)

        query, keys, values = (
            query.transpose(0, 2, 1),
            keys.transpose(0, 2, 1),
            values.transpose(0, 2, 1),
        )

        # attention
        attention = (query / self.embed_size_pow) @ keys.transpose(0, 1, 3, 2)
        attention = attention @ values

        if mask is not None:
            mask = mask.unsqueeze(1)
            attention[mask == 0] = -1e9

        attention, deriv = self.softmax.forward(attention, dim=-1)
        derivatives.append(deriv)

        attention, deriv = self.dropout1.forward(attention)
        derivatives.append(deriv)

        query = query.transpose(0, 2, 1).reshape(sz_b, len_q, -1)
        query, deriv = self.dropout2.forward(self.out_fc.forward(query))
        derivatives.append(deriv)
        query += residual

        query, deriv = self.norm.forward(query)
        derivatives.append(deriv)

        return query, attention, derivatives
