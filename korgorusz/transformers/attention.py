# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
"""
Implementation of attention from Transformers
"""
import numpy as np
from korgorusz.layers import LayerNorm, Linear, Dropout, Softmax, Array


class Attention:
    """
    'Attention' associate words...
    """

    def __init__(self, embed_size_pow: float, dropout_rate: float = 0.1):
        self.embed_size_pow = embed_size_pow
        self.dropout = Dropout(dropout_rate)
        self.softmax = Softmax()

    def forward(self, queries: Array, keys: Array, values: Array, mask: Array = None):
        """
        Calculates queries,keys and values.
        """
        derivatives = []
        attention = (queries / self.embed_size_pow) @ keys.transpose(0, 1, 3, 2)

        if mask is not None:
            attention[mask == 0] = -1e9

        attention, deriv = self.softmax.forward(attention, dim=-1)
        derivatives.append(deriv)

        attention, deriv = self.dropout.forward(attention)
        derivatives.append(deriv)
        output = attention @ values

        return output, attention, derivatives


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

        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNorm(d_model)
        self.attention = Attention(d_keys ** 0.5)

        self.values_fc = Linear(d_model, heads * d_keys, bias=False)
        self.keys_fc = Linear(d_model, heads * d_keys, bias=False)
        self.queries_fc = Linear(d_model, heads * d_values, bias=False)
        self.out_fc = Linear(heads * d_values, d_model, bias=False)

    def forward(self, query: Array, keys: Array, values: Array, mask: Array = None):
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
        if mask is not None:
            mask = np.expand_dims(mask, axis=1)

        attention = self.attention.forward(query, keys, values, mask=mask)

        query = query.transpose(0, 2, 1).reshape(sz_b, len_q, -1)
        query, deriv = self.dropout.forward(self.out_fc.forward(query))
        derivatives.append(deriv)
        query += residual

        query, deriv = self.norm.forward(query)
        derivatives.append(deriv)

        return query, attention, derivatives
