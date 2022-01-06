"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn

from config import *
import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.embeddings import PositionalEncoding
from onmt.modules.self_attention import SelfAttention
from onmt.modules.global_attention import GlobalAttention


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

p
q
u
            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask)
        sent = self.dropout(context) + inputs

        return self.feed_forward(sent)



class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.position_embeddings = PositionalEncoding(dropout, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.globAttn = GlobalAttention(d_model)
        # Add pro
        # single train:
        #self.pro_list = torch.tensor([[6], [7], [11], [64], [23], [62], [18], [15], [344], [67], [50002], [50003], [50004], [50005], [50006], [50007], [50008], [50009], [50010], [50011], [50012], [50013], [50014], [50015], [50016], [50017], [50018], [50019], [50020], [50021], [50022], [13], [68], [652], [789], [114], [196], [20], [170], [28], [97], [57], [343], [201], [123], [50023]]).cuda()
        #self.placeholder = torch.tensor([5]).cuda()
        # two_step train 46 pro:
        # self.pro_list = torch.tensor(
        #     [[10], [13], [121], [762], [60], [303], [252], [66], [2256], [645], [50002], [50003], [50004], [50005],
        #      [50006], [50007], [50008], [50009], [50010], [50011], [50012], [50013], [50014], [50015], [50016], [50017],
        #      [50018], [50019], [50020], [50021], [50022], [24], [399], [5221], [5768], [964], [2172], [130], [2160],
        #      [86], [554], [222], [2800], [1223], [784], [50023]]).cuda()
        # self.placeholder = torch.tensor([43]).cuda()
        # two_step train < 46 pro:
        self.pro_list = torch.tensor(
            [[10], [13], [121], [762], [60], [303], [252], [66], [2256], [645], [50002], [50003], [50004], [50005],
             [50006], [50007], [50008], [50009], [50010], [50011], [50012], [50013], [50014], [50015], [50016], [50017],
             [50018], [50019], [50020], [50021], [50022], [50023]]).cuda()
        self.placeholder = torch.tensor([43]).cuda()
        self.softmax = nn.Softmax()
        self.u1 = nn.Parameter(torch.empty(d_model, d_model, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, dtype=torch.float))

    def forward(self, src, lengths=None, freeze_d=True):
        """
            See :obj:`EncoderBase.forward()`
            first_step: freeze_d = True
            second_step: freeze_d = False
        """
        self._check_args(src, lengths)
        emb = self.embeddings(src)
        src_len = src.size(0)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)

        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        pro_h = None
        if not freeze_d:

            # pro embeddings:
            pro_emb = self.embeddings(self.pro_list.unsqueeze(0))  # (1, 46, 512)

            # Add pro mask:
            placeholder_ = self.placeholder.expand(words.size())
            pro_mask = words.data.eq(placeholder_.data).unsqueeze(-1).float()

            with torch.no_grad():
                out = out

            # discourse level input:
            sent = out
            per_sent_context, _ = self.self_attn(sent, sent, sent, mask=None)
            sent_context = torch.sum(per_sent_context, 1).unsqueeze(1)

            sent_context = self.dropout(sent_context)
            sent_context = self.position_embeddings(sent_context)

            for i in [0, 1, 2, 3, 4, 5]:
                sent_context = self.transformer[i](sent_context, None)
            disc_out = self.layer_norm(sent_context)  

            if Distribute:
                # discourse context distribute:
                # discourse level attn:
                disc_attn_context, disc_attn = self.context_attn(sent, sent, disc_out, mask=None)
                disc_attn = disc_attn.transpose(1, 2)
                c = torch.bmm(disc_attn, disc_out)  
            elif balance:
                c = disc_out.matmul(self.u1).mul(sent) + self.b
            else:
                c = disc_out.repeat(1, sent.size(1), 1) + sent

            #  pro score:
            pro_scores = self.globAttn.score(c, pro_emb)  
            pro_scores = self.softmax(pro_scores)

            # max:
            (pro_max, pro_max_index) = torch.max(pro_scores, dim=-1)  
            pro_max_index = pro_max_index.view(-1).long()
            pro_select = torch.index_select(pro_emb.squeeze(0), 0, pro_max_index)  
            pro_c = torch.mul(pro_select.view(c.size(0), c.size(1), -1), pro_max.unsqueeze(-1))  
            pro_h = pro_c.mul(pro_mask)  

            # pro replace:
            pro_predict = self.pro_list[pro_max_index].view(words.size(0), -1).mul(pro_mask.squeeze(-1))
            pro_predict = pro_predict*60000 + words.mul(1 - pro_mask.squeeze(-1))

            if replace_sentence:
                sent_ = sent.mul(1-pro_mask)
                sent_out = pro_h + sent_
                sent = sent_out
            if replace_discourse:
                c_ = c.mul(1 - pro_mask) 
                disc_out = pro_h + c_
                c = disc_out

            # with dropout (default)
            disc = self.dropout(c)
            # without dropout
            out = disc + sent

            return emb, out.transpose(0, 1).contiguous(), pro_h, pro_predict

