import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize
from transformers import BertModel, BertConfig
from sentence_transformer_wrapper import SentenceTransformerWrapper

INI = 1e-2
MAX_EXT = 6
CLS_ID=101
SEP_ID=102

class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None
        self._output_size = 3 * n_hidden

    def forward(self, input_):
        # input_:  [batch, seq_len]
        #print("conv_size")
        #print(input_.size())
        batch_size, seq_len = input_.size()
        if seq_len <= 2:
            left_padding = input_.new_zeros(batch_size, 2)
            right_padding = input_.new_zeros(batch_size, 2)
            input_ = torch.cat([left_padding, input_, right_padding], dim=1)
        elif seq_len <= 4:
            left_padding = input_.new_zeros(batch_size, 1)
            right_padding = input_.new_zeros(batch_size, 1)
            input_ = torch.cat([left_padding, input_, right_padding], dim=1)
        emb_input = self._embedding(input_) # [batch, seq_len, embed_size]
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)  # [batch, embed_size, seq_len]
        try:
            output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        except:
            print(input_.size())
            print(emb_input.size())
            print(conv_in.size())
            exit()
        #print("conv_out_size")
        #print(output.size())
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

    @property
    def output_size(self):
        return self._output_size


class ConvSentEncoderNoEmbedding(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, emb_dim, n_hidden, dropout):
        super().__init__()
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None
        self._output_size = 3 * n_hidden

    def forward(self, emb_input):
        # emb_input:  [batch, seq_len, embed_size]
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)  # [batch, embed_size, seq_len]
        try:
            output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        except:
            print(emb_input.size())
            print(conv_in.size())
            exit()
        return output

    @property
    def output_size(self):
        return self._output_size


class ConvCandidateAggregator(nn.Module):
    """
    Convolutional Candidate Aggregator
    w/ max-over-time pooling, [3] kernel sizes, ReLU activation
    """
    def __init__(self, input_dim, n_hidden, dropout):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, n_hidden, 3)
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        """
        :param input_: [batch, num_candidates, input_dim]
        :return:
        """
        conv_in = F.dropout(input_.transpose(1, 2),
                            self._dropout, training=self.training)  # [batch, num_candidates, input_dim]
        output = F.relu(self.conv(conv_in)).max(dim=2)[0]
        return output


class FCAggregator(nn.Module):
    """
    FC Candidate Aggregator
    """
    def __init__(self, num_candidates):
        super().__init__()
        self._fc = nn.Linear(num_candidates, 1, bias=False)
        self._fc.weight.data.fill_(1.0/num_candidates)  # init it to be mean pooling

    def forward(self, input_):
        """
        :param input_: [batch, num_candidates, input_dim]
        :return:
        """
        #fc_in = F.dropout(input_.transpose(1, 2), self._dropout, training=self.training)  # [batch, input_dim, num_candidates]
        fc_in = input_.transpose(1, 2)  # [batch, input_dim, num_candidates]
        output = self._fc(fc_in)  # [batch, input_dim, 1]
        return output.squeeze(2)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)  # [num_sents * num_]

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


class ExtractSumm(nn.Module):
    """ ff-ext """
    def __init__(self, vocab_size, emb_dim,
                 conv_hidden, lstm_hidden, lstm_layer,
                 bidirectional, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._sent_linear = nn.Linear(lstm_out_dim, 1)
        self._art_linear = nn.Linear(lstm_out_dim, lstm_out_dim)

    def forward(self, article_sents, sent_nums):
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        saliency = torch.cat(
            [s[:n] for s, n in zip(saliency, sent_nums)], dim=0)
        content = self._sent_linear(
            torch.cat([s[:n] for s, n in zip(enc_sent, sent_nums)], dim=0)
        )
        logit = (content + saliency).squeeze(1)
        return logit

    def extract(self, article_sents, sent_nums=None, k=4):
        """ extract top-k scored sentences from article (eval only)"""
        enc_sent, enc_art = self._encode(article_sents, sent_nums)
        saliency = torch.matmul(enc_sent, enc_art.unsqueeze(2))
        content = self._sent_linear(enc_sent)
        logit = (content + saliency).squeeze(2)
        if sent_nums is None:  # test-time extract only
            assert len(article_sents) == 1
            n_sent = logit.size(1)
            extracted = logit[0].topk(
                k if k < n_sent else n_sent, sorted=False  # original order
            )[1].tolist()
        else:
            extracted = [l[:n].topk(k if k < n else n)[1].tolist()
                         for n, l in zip(sent_nums, logit)]
        return extracted

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time extract only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)],
                           dim=0) if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        enc_art = F.tanh(
            self._art_linear(sequence_mean(lstm_out, sent_nums, dim=1)))
        return lstm_out, enc_art

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, auto_stop=False):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop
        self._auto_stop = auto_stop
        if auto_stop:
            self._stop = nn.Parameter(torch.Tensor(input_dim))
            init.uniform_(self._stop, -INI, INI)

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        if self._auto_stop:
            batch_size, max_sent_num, input_dim = attn_mem.size()
            # insert stop representation
            attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)  # [batch, max_sent_num+1, input_dim]
            mem_sizes_tensor = torch.LongTensor(mem_sizes).to(attn_mem.device)
            attn_mem[:, mem_sizes_tensor, :] = self._stop.unsqueeze(0).expand(batch_size, -1)  # [batch_size, input_dim]
            mem_sizes = [s+1 for s in mem_sizes]

        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        if lstm_in is not None:
            lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        else:
            lstm_in = init_i.transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k, disable_selected_mask=False):
        """extract k sentences, decode only, batch_size==1"""
        # atten_mem: Tensor of size [1, max_sent_num, input_dim]
        if self._auto_stop:
            end_step = attn_mem.size(1)
            attn_mem = torch.cat([attn_mem, self._stop.view(1, 1, -1)], dim=1)  # [1, max_sent_num+1, input_dim]

        use_selected_mask = not disable_selected_mask
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        num_extracted_sent = 0
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)  # [1, 1, Ns]
            score = score.squeeze()  # [Ns]
            # set logit to -inf if the sentence is selected before
            if use_selected_mask:
                for e in extracts:
                    score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            if self._auto_stop:  # break the loop if eos is selected, does not include eos to the extracts
                if ext == end_step:
                    break
            extracts.append(ext)
            num_extracted_sent += 1
            if (not self._auto_stop and num_extracted_sent == k) or num_extracted_sent == MAX_EXT:
                break
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output


class LSTMPointerNetWithCandidate(LSTMPointerNet):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer, dropout, n_hop, auto_stop=False):
        super().__init__(input_dim, n_hidden, n_layer, dropout, n_hop, auto_stop)

    def extract(self, attn_mem, mem_sizes, k, num_candidates=1, disable_selected_mask=False):
        """extract k sentences, decode only, batch_size==1"""
        # atten_mem: Tensor of size [1, max_sent_num, input_dim]
        if self._auto_stop:
            end_step = attn_mem.size(1)
            attn_mem = torch.cat([attn_mem, self._stop.view(1, 1, -1)], dim=1)  # [1, max_sent_num+1, input_dim]

        use_selected_mask = not disable_selected_mask
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        num_extracted_sent = 0
        masked_candidates = []
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)  # [1, 1, Ns]
            score = score.squeeze()  # [Ns]

            # set logit to -inf if the sentence is selected before
            for e in masked_candidates:
                score[e] = -1e6

            ext = score.max(dim=0)[1].item()
            if self._auto_stop:  # break the loop if eos is selected, does not include eos to the extracts
                if ext == end_step:
                    break
            extracts.append(ext)
            num_extracted_sent += 1
            if (not self._auto_stop and num_extracted_sent == k) or num_extracted_sent == MAX_EXT:
                break
            # map the extracted candidate idx to all the candidate indices in that sentence and mask them out
            selected_sent_idx = ext // num_candidates
            if use_selected_mask:
                masked_candidates += [selected_sent_idx * num_candidates + i for i in range(num_candidates)]

            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0, auto_stop=False):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, auto_stop
        )

    def forward(self, article_sents, sent_nums, target):
        enc_out = self._encode(article_sents, sent_nums)
        if target is not None:
            bs, nt = target.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(
                enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
            )
        else:
            ptr_in = None
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    def extract(self, article_sents, sent_nums=None, k=4, disable_selected_mask=False):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k, disable_selected_mask)
        return output

    def _encode(self, article_sents, sent_nums):
        """
        :param article_sents:
        :param sent_nums: a list of sent_nums
        :return:
        """
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]  # each item has dimension [num_sents, 3*conv_size]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            # pad the article that with sent_num less than max_n
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)


class PtrExtractRewrittenSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 num_candidates, n_hop=1, dropout=0.0, candidate_agg_type='mean', auto_stop=False):
        super().__init__()
        self._candidate_sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        #self._candidate_agg  = ConvCandidateAggregator(3*conv_hidden, 3*conv_hidden, dropout)
        if candidate_agg_type == 'mean':
            self._candidate_agg = lambda x: torch.mean(x, dim=1)  # mean pooling over the candidate representations
        else:
            self._candidate_agg = FCAggregator(num_candidates)

        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1) + 3*conv_hidden
        self._extractor = LSTMPointerNetWithCandidate(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, auto_stop
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.num_candidates = num_candidates

    def forward(self, article_sents, total_cand_nums, target):
        sent_nums = [cand_num//self.num_candidates for cand_num in total_cand_nums]

        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]
        # debug
        """
        print("global encode_out size: {}".format(context_aware_encode_out.size()))
        print("local encode_out size: {}".format(local_encode_out.size()))
        print("local_encode[0, 0, 0, 0]: {}".format(local_encode_out[0, 0, 0, 0:5]))
        print("local_encode[0, 0, 1, 0]: {}".format(local_encode_out[0, 0, 1, 0:5]))
        print("local_encode[0, 1, 0, 0]: {}".format(local_encode_out[0, 1, 0, 0:5]))
        print("local_encode[0, 1, 1, 0]: {}".format(local_encode_out[0, 1, 1, 0:5]))
        print("local_encode[1, 0, 0, 0]: {}".format(local_encode_out[1, 0, 0, 0:5]))
        print("local_encode[1, 0, 1, 0]: {}".format(local_encode_out[1, 0, 1, 0:5]))
        print("global_encode[0, 0, 0]: {}".format(context_aware_encode_out[0, 0, 0:5]))
        print("global_encode[0, 1, 0]: {}".format(context_aware_encode_out[0, 1, 0:5]))
        print("global_encode[1, 0, 0]: {}".format(context_aware_encode_out[1, 0, 0:5]))
        """

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out, context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates, lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        # debug
        """
        print("enc_out[0, 0, 0]: {}".format(enc_out[0, 0, 0:5]))
        print("enc_out[0, 0, 300]: {}".format(enc_out[0, 0, 300:305]))
        print("enc_out[0, 1, 0]: {}".format(enc_out[0, 1, 0:5]))
        print("enc_out[0, 1, 300]: {}".format(enc_out[0, 1, 300:305]))
        print("enc_out[0, 2, 0]: {}".format(enc_out[0, 2, 0:5]))
        print("enc_out[0, 2, 300]: {}".format(enc_out[0, 2, 300:305]))
        print("enc_out[0, 3, 0]: {}".format(enc_out[0, 3, 0:5]))
        print("enc_out[0, 3, 300]: {}".format(enc_out[0, 3, 300:305]))
        print("enc_out[1, 0, 0]: {}".format(enc_out[1, 0, 0:5]))
        print("enc_out[1, 0, 300]: {}".format(enc_out[1, 0, 300:305]))
        print("enc_out[1, 1, 0]: {}".format(enc_out[1, 1, 0:5]))
        print("enc_out[1, 1, 300]: {}".format(enc_out[1, 1, 300:305]))
        print()
        print("enc out size: {}".format(enc_out.size()))
        """
        if target is not None:
            batch_size, nt = target.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(
                enc_out, dim=1, index=target.unsqueeze(2).expand(batch_size, nt, d)
            )  # [batch, nt, d]
        else:
            ptr_in = None

        #total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor(enc_out, total_cand_nums, ptr_in)

        return output

    def extract(self, article_sents, total_cand_nums=None, k=4, disable_selected_mask=False):
        if total_cand_nums is not None:
            sent_nums = [n // self.num_candidates for n in total_cand_nums]
        else:
            sent_nums = None
        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out,
                             context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates,
                                                                          lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        #total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor.extract(enc_out, total_cand_nums, k, self.num_candidates, disable_selected_mask)
        return output

    def _encode(self, article_sents, sent_nums):
        """
        :param article_sents: a list of tensor, each tensor has dim [num_sent * num_candidates, seq_len]
        :param sent_nums: a list of sent_nums
        :return: lstm_out: [batch, max_n, 2 * lstm_hidden], enc_candidates_out: [batch, max_n, num_cands, 3*conv_size]
        """
        batch_size = len(article_sents)
        if sent_nums is None:  # test-time excode only
            enc_candidates = self._candidate_sent_enc(article_sents[0]).view(article_sents[0].size(0)//self.num_candidates, self.num_candidates, -1)
            # [num_sents, num_cands, 3*conv_size]
            enc_sents = self._candidate_agg(enc_candidates)  # [num_sents, 3*conv_size]
            enc_sents_padded = enc_sents.unsqueeze(0)  # [1, num_sents, 3*conv_size]
            enc_candidates_out = enc_candidates.unsqueeze(0)  # [1, num_sents, num_cands, 3*conv_size]

        else:
            max_n = max(sent_nums)
            enc_candidates = [self._candidate_sent_enc(art_sent).view(art_sent.size(0)//self.num_candidates, self.num_candidates, -1)
                         for art_sent in article_sents]  # each item has dimension [num_sents, num_cands, 3*conv_size]
            enc_candidates_flattened = torch.cat(enc_candidates, dim=0)  # [total_num_sents_in_batch, num_cands, 3*conv_size]
            enc_sents_flattened = self._candidate_agg(enc_candidates_flattened)  # [total_num_sents_in_batch, 3*conv_size]


            # pad an article if its sent_num less than max_n
            def zero_enc_sent(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            start_idx = 0
            enc_sents_padded = []
            for n in sent_nums:
                end_idx = start_idx + n
                s = enc_sents_flattened[start_idx:end_idx, :]
                padded_s = torch.cat([s, zero_enc_sent(max_n-n, s.device)], dim=0) if n != max_n else s  # [max_n, 3*conv_size]
                enc_sents_padded.append(padded_s)
                start_idx = end_idx

            # pad enc_sentences to max_n
            enc_sents_padded = torch.stack(enc_sents_padded, dim=0).contiguous()  # [batch, max_n, 3*conv_size]

            # pad enc_candidates to max_n
            def zero_enc_candidate(n, device):
                z = torch.zeros(n, self.num_candidates, self._candidate_sent_enc.output_size).to(device)
                return z

            # a list of [num_sents, num_cands, 3*conv_size] -> [batch, max_n, num_cands, 3*conv_size]
            enc_candidates_out = torch.stack([torch.cat([s, zero_enc_candidate(max_n - n, s.device)], dim=0)
                                              if n != max_n else s for s, n in zip(enc_candidates, sent_nums)],
                                             dim=0)  # [batch, max_n, num_cands, 3*conv_size]

        # compute context-aware embedding for each article sentence
        lstm_out = self._art_enc(enc_sents_padded, sent_nums)  # [batch, max_n, 2 * lstm_hidden]

        return lstm_out, enc_candidates_out

    def set_embedding(self, embedding):
        self._candidate_sent_enc.set_embedding(embedding)


class PtrExtractRewrittenSentBertSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 num_candidates, cache_dir, n_hop=1, dropout=0.0, candidate_agg_type='mean', auto_stop=False):
        super().__init__()
        self.num_bert_layers = 3
        self._sentence_encoder = SentenceTransformerWrapper(model_name_or_path='bert-base-nli-mean-tokens', num_bert_layers=self.num_bert_layers)
        self._sentence_encoder.eval()
        for p in self._sentence_encoder.parameters():
            p.requires_grad = False
        self._bert_w = nn.Linear(self.num_bert_layers * 768, 3*conv_hidden)
        #self._candidate_agg  = ConvCandidateAggregator(3*conv_hidden, 3*conv_hidden, dropout)
        if candidate_agg_type == 'mean':
            self._candidate_agg = lambda x: torch.mean(x, dim=1)  # mean pooling over the candidate representations
        else:
            self._candidate_agg = FCAggregator(num_candidates)

        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1) + 3*conv_hidden
        self._extractor = LSTMPointerNetWithCandidate(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, auto_stop
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.num_candidates = num_candidates

    def forward(self, article_sents, total_cand_nums, target):
        sent_nums = [cand_num//self.num_candidates for cand_num in total_cand_nums]

        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]
        # debug
        """
        print("global encode_out size: {}".format(context_aware_encode_out.size()))
        print("local encode_out size: {}".format(local_encode_out.size()))
        print("local_encode[0, 0, 0, 0]: {}".format(local_encode_out[0, 0, 0, 0:5]))
        print("local_encode[0, 0, 1, 0]: {}".format(local_encode_out[0, 0, 1, 0:5]))
        print("local_encode[0, 1, 0, 0]: {}".format(local_encode_out[0, 1, 0, 0:5]))
        print("local_encode[0, 1, 1, 0]: {}".format(local_encode_out[0, 1, 1, 0:5]))
        print("local_encode[1, 0, 0, 0]: {}".format(local_encode_out[1, 0, 0, 0:5]))
        print("local_encode[1, 0, 1, 0]: {}".format(local_encode_out[1, 0, 1, 0:5]))
        print("global_encode[0, 0, 0]: {}".format(context_aware_encode_out[0, 0, 0:5]))
        print("global_encode[0, 1, 0]: {}".format(context_aware_encode_out[0, 1, 0:5]))
        print("global_encode[1, 0, 0]: {}".format(context_aware_encode_out[1, 0, 0:5]))
        """

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out, context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates, lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        # debug
        """
        print("enc_out[0, 0, 0]: {}".format(enc_out[0, 0, 0:5]))
        print("enc_out[0, 0, 300]: {}".format(enc_out[0, 0, 300:305]))
        print("enc_out[0, 1, 0]: {}".format(enc_out[0, 1, 0:5]))
        print("enc_out[0, 1, 300]: {}".format(enc_out[0, 1, 300:305]))
        print("enc_out[0, 2, 0]: {}".format(enc_out[0, 2, 0:5]))
        print("enc_out[0, 2, 300]: {}".format(enc_out[0, 2, 300:305]))
        print("enc_out[0, 3, 0]: {}".format(enc_out[0, 3, 0:5]))
        print("enc_out[0, 3, 300]: {}".format(enc_out[0, 3, 300:305]))
        print("enc_out[1, 0, 0]: {}".format(enc_out[1, 0, 0:5]))
        print("enc_out[1, 0, 300]: {}".format(enc_out[1, 0, 300:305]))
        print("enc_out[1, 1, 0]: {}".format(enc_out[1, 1, 0:5]))
        print("enc_out[1, 1, 300]: {}".format(enc_out[1, 1, 300:305]))
        print()
        print("enc out size: {}".format(enc_out.size()))
        """
        if target is not None:
            batch_size, nt = target.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(
                enc_out, dim=1, index=target.unsqueeze(2).expand(batch_size, nt, d)
            )  # [batch, nt, d]
        else:
            ptr_in = None

        #total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor(enc_out, total_cand_nums, ptr_in)

        return output

    def extract(self, article_sents, total_cand_nums=None, k=4, disable_selected_mask=False):
        if total_cand_nums is not None:
            sent_nums = [n // self.num_candidates for n in total_cand_nums]
        else:
            sent_nums = None
        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out,
                             context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates,
                                                                          lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        #total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor.extract(enc_out, total_cand_nums, k, self.num_candidates, disable_selected_mask)
        return output

    def _encode(self, article_sents, sent_nums):
        """
        :param article_sents: a list of tensor, each tensor has dim [num_sent * num_candidates, seq_len]
        :param sent_nums: a list of sent_nums
        :return: lstm_out: [batch, max_n, 2 * lstm_hidden], enc_candidates_out: [batch, max_n, num_cands, 3*conv_size]
        """
        batch_size = len(article_sents)
        if sent_nums is None:  # test-time excode only
            enc_candidates = self._candidate_sent_encode(article_sents[0]).view(article_sents[0].size(0)//self.num_candidates, self.num_candidates, -1)
            # [num_sents, num_cands, 3*conv_size]
            enc_sents = self._candidate_agg(enc_candidates)  # [num_sents, 3*conv_size]
            enc_sents_padded = enc_sents.unsqueeze(0)  # [1, num_sents, 3*conv_size]
            enc_candidates_out = enc_candidates.unsqueeze(0)  # [1, num_sents, num_cands, 3*conv_size]

        else:
            max_n = max(sent_nums)
            enc_candidates = [self._candidate_sent_encode(art_sent).view(art_sent.size(0)//self.num_candidates, self.num_candidates, -1)
                         for art_sent in article_sents]  # each item has dimension [num_sents, num_cands, 3*conv_size]
            enc_candidates_flattened = torch.cat(enc_candidates, dim=0)  # [total_num_sents_in_batch, num_cands, 3*conv_size]
            enc_sents_flattened = self._candidate_agg(enc_candidates_flattened)  # [total_num_sents_in_batch, 3*conv_size]


            # pad an article if its sent_num less than max_n
            def zero_enc_sent(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            start_idx = 0
            enc_sents_padded = []
            for n in sent_nums:
                end_idx = start_idx + n
                s = enc_sents_flattened[start_idx:end_idx, :]
                padded_s = torch.cat([s, zero_enc_sent(max_n-n, s.device)], dim=0) if n != max_n else s  # [max_n, 3*conv_size]
                enc_sents_padded.append(padded_s)
                start_idx = end_idx

            # pad enc_sentences to max_n
            enc_sents_padded = torch.stack(enc_sents_padded, dim=0).contiguous()  # [batch, max_n, 3*conv_size]

            # pad enc_candidates to max_n
            def zero_enc_candidate(n, device):
                z = torch.zeros(n, self.num_candidates, enc_candidates[0].size(2)).to(device)
                return z

            # a list of [num_sents, num_cands, 3*conv_size] -> [batch, max_n, num_cands, 3*conv_size]
            enc_candidates_out = torch.stack([torch.cat([s, zero_enc_candidate(max_n - n, s.device)], dim=0)
                                              if n != max_n else s for s, n in zip(enc_candidates, sent_nums)],
                                             dim=0)  # [batch, max_n, num_cands, 3*conv_size]

        # compute context-aware embedding for each article sentence
        lstm_out = self._art_enc(enc_sents_padded, sent_nums)  # [batch, max_n, 2 * lstm_hidden]

        return lstm_out, enc_candidates_out

    def _candidate_sent_encode(self, sents_tensor):
        # sents_tensor: [num_sent * num_candidates, seq_len]
        #num_sents_all, sent_len = sents_tensor.size()

        attention_mask_tensor = torch.ne(sents_tensor, 0).to(sents_tensor.device)
        sent_bert_embeddings = self._sentence_encoder.encode_tensor(sents_tensor, attention_mask_tensor)
        # [num_sents * num_cands, 768 * num_bert_layers]

        sent_embeddings = self._bert_w(sent_bert_embeddings)  # [num_sent * num_candidates, 3 * conv_size]
        #enc_candidate = sent_embeddings.view(num_sents_all // self.num_candidates, self.num_candidates, -1)
        # [num_doc_sents, num_cands, 3 * conv_size]
        #print("sent_embeedings size")
        #print(sent_embeddings.size())
        #exit()
        return sent_embeddings


class PtrExtractRewrittenBertSumm(nn.Module):
    """ rnn-ext"""

    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 num_candidates, cache_dir, n_hop=1, dropout=0.0, candidate_agg_type='mean', auto_stop=False):
        super().__init__()
        #model_name_or_path = "bert-large-uncased"
        model_name_or_path = "bert-base-uncased"
        config = BertConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        config.output_hidden_states = True
        self.bert_model = BertModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.bert_model.eval()
        for p in self.bert_model.parameters():
            p.requires_grad = False
        #self._bert_w = nn.Linear(1024 * 4, emb_dim)
        self._bert_w = nn.Linear(768 * 4, emb_dim)
        self._candidate_sent_enc = ConvSentEncoderNoEmbedding(emb_dim, conv_hidden, dropout)
        # self._candidate_agg  = ConvCandidateAggregator(3*conv_hidden, 3*conv_hidden, dropout)
        if candidate_agg_type == 'mean':
            self._candidate_agg = lambda x: torch.mean(x, dim=1)  # mean pooling over the candidate representations
        else:
            self._candidate_agg = FCAggregator(num_candidates)

        self._art_enc = LSTMEncoder(
            3 * conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1) + 3 * conv_hidden
        self._extractor = LSTMPointerNetWithCandidate(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, auto_stop
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.num_candidates = num_candidates

    def forward(self, article_sents, total_cand_nums, target):
        sent_nums = [cand_num // self.num_candidates for cand_num in total_cand_nums]

        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]
        # debug
        """
        print("global encode_out size: {}".format(context_aware_encode_out.size()))
        print("local encode_out size: {}".format(local_encode_out.size()))
        print("local_encode[0, 0, 0, 0]: {}".format(local_encode_out[0, 0, 0, 0:5]))
        print("local_encode[0, 0, 1, 0]: {}".format(local_encode_out[0, 0, 1, 0:5]))
        print("local_encode[0, 1, 0, 0]: {}".format(local_encode_out[0, 1, 0, 0:5]))
        print("local_encode[0, 1, 1, 0]: {}".format(local_encode_out[0, 1, 1, 0:5]))
        print("local_encode[1, 0, 0, 0]: {}".format(local_encode_out[1, 0, 0, 0:5]))
        print("local_encode[1, 0, 1, 0]: {}".format(local_encode_out[1, 0, 1, 0:5]))
        print("global_encode[0, 0, 0]: {}".format(context_aware_encode_out[0, 0, 0:5]))
        print("global_encode[0, 1, 0]: {}".format(context_aware_encode_out[0, 1, 0:5]))
        print("global_encode[1, 0, 0]: {}".format(context_aware_encode_out[1, 0, 0:5]))
        """

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out,
                             context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates,
                                                                          lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates,
                               -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        # debug
        """
        print("enc_out[0, 0, 0]: {}".format(enc_out[0, 0, 0:5]))
        print("enc_out[0, 0, 300]: {}".format(enc_out[0, 0, 300:305]))
        print("enc_out[0, 1, 0]: {}".format(enc_out[0, 1, 0:5]))
        print("enc_out[0, 1, 300]: {}".format(enc_out[0, 1, 300:305]))
        print("enc_out[0, 2, 0]: {}".format(enc_out[0, 2, 0:5]))
        print("enc_out[0, 2, 300]: {}".format(enc_out[0, 2, 300:305]))
        print("enc_out[0, 3, 0]: {}".format(enc_out[0, 3, 0:5]))
        print("enc_out[0, 3, 300]: {}".format(enc_out[0, 3, 300:305]))
        print("enc_out[1, 0, 0]: {}".format(enc_out[1, 0, 0:5]))
        print("enc_out[1, 0, 300]: {}".format(enc_out[1, 0, 300:305]))
        print("enc_out[1, 1, 0]: {}".format(enc_out[1, 1, 0:5]))
        print("enc_out[1, 1, 300]: {}".format(enc_out[1, 1, 300:305]))
        print()
        print("enc out size: {}".format(enc_out.size()))
        """
        if target is not None:
            batch_size, nt = target.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(
                enc_out, dim=1, index=target.unsqueeze(2).expand(batch_size, nt, d)
            )  # [batch, nt, d]
        else:
            ptr_in = None

        # total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor(enc_out, total_cand_nums, ptr_in)
        return output

    def extract(self, article_sents, total_cand_nums=None, k=4, disable_selected_mask=False):
        if total_cand_nums is not None:
            sent_nums = [n // self.num_candidates for n in total_cand_nums]
        else:
            sent_nums = None
        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out,
                             context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates,
                                                                          lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        #total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor.extract(enc_out, total_cand_nums, k, self.num_candidates, disable_selected_mask)
        return output

    def _encode(self, article_sents, sent_nums):
        """
        :param article_sents: a list of tensor, each tensor has dim [num_sent * num_candidates, seq_len]
        :param sent_nums: a list of sent_nums
        :return: lstm_out: [batch, max_n, 2 * lstm_hidden], enc_candidates_out: [batch, max_n, num_cands, 3*conv_size]
        """
        batch_size = len(article_sents)
        device = article_sents[0].device
        if sent_nums is None:  # test-time excode only
            enc_candidates = self._article_encode(article=article_sents[0], device=device)  # [num_sents, num_cands, 3*conv_size]
            enc_sents = self._candidate_agg(enc_candidates)  # [num_sents, 3*conv_size]
            enc_sents_padded = enc_sents.unsqueeze(0)  # [1, num_sents, 3*conv_size]
            enc_candidates_out = enc_candidates.unsqueeze(0)  # [1, num_sents, num_cands, 3*conv_size]
        else:
            max_n = max(sent_nums)
            enc_candidates = [self._article_encode(article=article, device=device) for article in article_sents]
            # enc_candidates: each item has size of [num_sents, num_cands, 3*conv_size]
            enc_candidates_flattened = torch.cat(enc_candidates, dim=0)  # [total_num_sents_in_batch, num_cands, 3*conv_size]
            #print()
            #print("enc_candidates_flattened.size")
            #print(enc_candidates_flattened.size())
            enc_sents_flattened = self._candidate_agg(enc_candidates_flattened)  # [total_num_sents_in_batch, 3*conv_size]
            #print("enc_sents_flattened.size()")
            #print(enc_sents_flattened.size())

            # pad an article if its sent_num less than max_n
            def zero_enc_sent(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            start_idx = 0
            enc_sents_padded = []
            for n in sent_nums:
                end_idx = start_idx + n
                s = enc_sents_flattened[start_idx:end_idx, :]
                padded_s = torch.cat([s, zero_enc_sent(max_n-n, s.device)], dim=0) if n != max_n else s  # [max_n, 3*conv_size]
                enc_sents_padded.append(padded_s)
                start_idx = end_idx

            # pad enc_sentences to max_n
            enc_sents_padded = torch.stack(enc_sents_padded, dim=0).contiguous()  # [batch, max_n, 3*conv_size]
            #print("enc_sents_padded_size")
            #print(enc_sents_padded.size())

            # pad enc_candidates to max_n
            def zero_enc_candidate(n, device):
                z = torch.zeros(n, self.num_candidates, self._candidate_sent_enc.output_size).to(device)
                return z

            # a list of [num_sents, num_cands, 3*conv_size] -> [batch, max_n, num_cands, 3*conv_size]
            enc_candidates_out = torch.stack([torch.cat([s, zero_enc_candidate(max_n - n, s.device)], dim=0)
                                              if n != max_n else s for s, n in zip(enc_candidates, sent_nums)],
                                             dim=0)  # [batch, max_n, num_cands, 3*conv_size]

        # compute context-aware embedding for each article sentence
        lstm_out = self._art_enc(enc_sents_padded, sent_nums)  # [batch, max_n, 2 * lstm_hidden]
        #print("lstm_out.size()")
        #print(lstm_out.size())
        #print("enc_candidates_out_size")
        #print(enc_candidates_out.size())

        return lstm_out, enc_candidates_out

    def _article_encode(self, article, device, pad_idx=0):
        # process one article
        # create input for BERT
        num_sents_all, sent_len = article.size()
        #print("article_size")
        #print(article.size())
        article_token_ids = [[CLS_ID] for _ in range(self.num_candidates)]  # 2d list [num_cands, art_len]
        article_input_mask = []
        article_cand_lens = [1] * self.num_candidates  # include cls and sep
        for i in range(num_sents_all):
            cand_i = i % self.num_candidates
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    article_token_ids[cand_i].append(article[i][j])
                    article_cand_lens[cand_i] += 1
                else:
                    break
        for cand_i in range(self.num_candidates):
            article_token_ids[cand_i].append(SEP_ID)
            article_cand_lens[cand_i] += 1
            article_input_mask.append([1] * article_cand_lens[cand_i])

        # compute max seq_len among all cands
        max_article_len = max(article_cand_lens)

        # padding
        for cand_i in range(self.num_candidates):
            while len(article_token_ids[cand_i]) < max_article_len:
                article_token_ids[cand_i].append(0)
                article_input_mask[cand_i].append(0)
            assert len(article_token_ids[cand_i]) == max_article_len
            assert len(article_input_mask[cand_i]) == max_article_len

        bert_attention_mask = torch.LongTensor(article_input_mask).to(device)

        bert_input_ids = torch.LongTensor(article_token_ids).to(device)

        last_hidden_state, _, hidden_states = self.bert_model(input_ids=bert_input_ids,
                                                              attention_mask=bert_attention_mask, token_type_ids=None)

        bert_emb_out = torch.cat([hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]], dim=-1)
        assert bert_emb_out.size() == (self.num_candidates, max_article_len, 3072)
        #assert bert_emb_out.size() == (self.num_candidates, max_article_len, 4096)

        emb_out = self._bert_w(bert_emb_out)  # [num_cands, max_article_len, 128]
        #print("emb_out")
        #print(emb_out.size())

        emb_dim = emb_out.size(-1)  # 128

        emb_input = torch.zeros(num_sents_all, sent_len, emb_dim).to(device)
        cur_ids = [1] * self.num_candidates  # after [CLS]
        for i in range(num_sents_all):
            cand_i = i % self.num_candidates
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    emb_input[i][j] = emb_out[cand_i][cur_ids[cand_i]]
                    cur_ids[cand_i] += 1
                else:
                    break

        for cand_i in range(self.num_candidates):
            assert cur_ids[cand_i] == article_cand_lens[cand_i] - 1
        #print()

        enc_candidate = self._candidate_sent_enc(emb_input)
        try:
            enc_candidate = enc_candidate.view(num_sents_all // self.num_candidates,self.num_candidates, -1)
        except:
            print("emb_input size")
            print(emb_input.size())
            print("num_sents_all")
            print(num_sents_all)
            print("emb_out")
            print(emb_out.size())
            print("enc_candidate.size()")
            print(enc_candidate.size())
            exit()
        # [num_doc_sents, num_cands, 3 * conv_size]
        #print("enc_candidate_size:")
        #print(enc_candidate.size())
        return enc_candidate


class PtrExtractRewrittenSentWordBertSumm(nn.Module):
    """ rnn-ext"""

    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 num_candidates, cache_dir, n_hop=1, dropout=0.0, candidate_agg_type='mean', auto_stop=False):
        super().__init__()
        # self.num_bert_layers = num_bert_layers
        self.num_bert_layers = 1
        self._sentence_encoder = SentenceTransformerWrapper(model_name_or_path='bert-base-nli-mean-tokens', num_bert_layers=self.num_bert_layers)
        self._sentence_encoder.eval()
        for p in self._sentence_encoder.parameters():
            p.requires_grad = False
        self._bert_w = nn.Linear(self.num_bert_layers * 768, emb_dim)
        self._candidate_sent_enc = ConvSentEncoderNoEmbedding(emb_dim, conv_hidden, dropout)
        # self._candidate_agg  = ConvCandidateAggregator(3*conv_hidden, 3*conv_hidden, dropout)
        if candidate_agg_type == 'mean':
            self._candidate_agg = lambda x: torch.mean(x, dim=1)  # mean pooling over the candidate representations
        else:
            self._candidate_agg = FCAggregator(num_candidates)

        self._art_enc = LSTMEncoder(
            3 * conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1) + 3 * conv_hidden
        self._extractor = LSTMPointerNetWithCandidate(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop, auto_stop
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.num_candidates = num_candidates

    def forward(self, article_sents, total_cand_nums, target):
        sent_nums = [cand_num // self.num_candidates for cand_num in total_cand_nums]

        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]
        # debug
        """
        print("global encode_out size: {}".format(context_aware_encode_out.size()))
        print("local encode_out size: {}".format(local_encode_out.size()))
        print("local_encode[0, 0, 0, 0]: {}".format(local_encode_out[0, 0, 0, 0:5]))
        print("local_encode[0, 0, 1, 0]: {}".format(local_encode_out[0, 0, 1, 0:5]))
        print("local_encode[0, 1, 0, 0]: {}".format(local_encode_out[0, 1, 0, 0:5]))
        print("local_encode[0, 1, 1, 0]: {}".format(local_encode_out[0, 1, 1, 0:5]))
        print("local_encode[1, 0, 0, 0]: {}".format(local_encode_out[1, 0, 0, 0:5]))
        print("local_encode[1, 0, 1, 0]: {}".format(local_encode_out[1, 0, 1, 0:5]))
        print("global_encode[0, 0, 0]: {}".format(context_aware_encode_out[0, 0, 0:5]))
        print("global_encode[0, 1, 0]: {}".format(context_aware_encode_out[0, 1, 0:5]))
        print("global_encode[1, 0, 0]: {}".format(context_aware_encode_out[1, 0, 0:5]))
        """

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out,
                             context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates,
                                                                          lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates,
                               -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        # debug
        """
        print("enc_out[0, 0, 0]: {}".format(enc_out[0, 0, 0:5]))
        print("enc_out[0, 0, 300]: {}".format(enc_out[0, 0, 300:305]))
        print("enc_out[0, 1, 0]: {}".format(enc_out[0, 1, 0:5]))
        print("enc_out[0, 1, 300]: {}".format(enc_out[0, 1, 300:305]))
        print("enc_out[0, 2, 0]: {}".format(enc_out[0, 2, 0:5]))
        print("enc_out[0, 2, 300]: {}".format(enc_out[0, 2, 300:305]))
        print("enc_out[0, 3, 0]: {}".format(enc_out[0, 3, 0:5]))
        print("enc_out[0, 3, 300]: {}".format(enc_out[0, 3, 300:305]))
        print("enc_out[1, 0, 0]: {}".format(enc_out[1, 0, 0:5]))
        print("enc_out[1, 0, 300]: {}".format(enc_out[1, 0, 300:305]))
        print("enc_out[1, 1, 0]: {}".format(enc_out[1, 1, 0:5]))
        print("enc_out[1, 1, 300]: {}".format(enc_out[1, 1, 300:305]))
        print()
        print("enc out size: {}".format(enc_out.size()))
        """
        if target is not None:
            batch_size, nt = target.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(
                enc_out, dim=1, index=target.unsqueeze(2).expand(batch_size, nt, d)
            )  # [batch, nt, d]
        else:
            ptr_in = None

        # total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor(enc_out, total_cand_nums, ptr_in)
        return output

    def extract(self, article_sents, total_cand_nums=None, k=4, disable_selected_mask=False):
        if total_cand_nums is not None:
            sent_nums = [n // self.num_candidates for n in total_cand_nums]
        else:
            sent_nums = None
        context_aware_encode_out, local_encode_out = self._encode(article_sents, sent_nums)
        # [batch, max_n, 2 * lstm_hidden], [batch, max_n, num_cands, 3*conv_size]

        batch_size, max_n, lstm_out_dim = context_aware_encode_out.size()
        # concat local representation with context-aware representation
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([local_encode_out,
                             context_aware_encode_out.unsqueeze(2).expand(batch_size, max_n, self.num_candidates,
                                                                          lstm_out_dim)], dim=3)
        enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)  # [batch, max_n * num_cand, 2*lstm_hidden + 3*conv_size]

        #total_cand_nums = [n * self.num_candidates for n in sent_nums]
        output = self._extractor.extract(enc_out, total_cand_nums, k, self.num_candidates, disable_selected_mask)
        return output

    def _encode(self, article_sents, sent_nums):
        """
        :param article_sents: a list of tensor, each tensor has dim [num_sent * num_candidates, seq_len]
        :param sent_nums: a list of sent_nums
        :return: lstm_out: [batch, max_n, 2 * lstm_hidden], enc_candidates_out: [batch, max_n, num_cands, 3*conv_size]
        """
        batch_size = len(article_sents)
        device = article_sents[0].device
        if sent_nums is None:  # test-time excode only
            enc_candidates = self._article_encode(article=article_sents[0], device=device)  # [num_sents, num_cands, 3*conv_size]
            enc_sents = self._candidate_agg(enc_candidates)  # [num_sents, 3*conv_size]
            enc_sents_padded = enc_sents.unsqueeze(0)  # [1, num_sents, 3*conv_size]
            enc_candidates_out = enc_candidates.unsqueeze(0)  # [1, num_sents, num_cands, 3*conv_size]
        else:
            max_n = max(sent_nums)
            enc_candidates = [self._article_encode(article=article, device=device) for article in article_sents]
            # enc_candidates: each item has size of [num_sents, num_cands, 3*conv_size]
            enc_candidates_flattened = torch.cat(enc_candidates, dim=0)  # [total_num_sents_in_batch, num_cands, 3*conv_size]
            #print()
            #print("enc_candidates_flattened.size")
            #print(enc_candidates_flattened.size())
            enc_sents_flattened = self._candidate_agg(enc_candidates_flattened)  # [total_num_sents_in_batch, 3*conv_size]
            #print("enc_sents_flattened.size()")
            #print(enc_sents_flattened.size())

            # pad an article if its sent_num less than max_n
            def zero_enc_sent(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z

            start_idx = 0
            enc_sents_padded = []
            for n in sent_nums:
                end_idx = start_idx + n
                s = enc_sents_flattened[start_idx:end_idx, :]
                padded_s = torch.cat([s, zero_enc_sent(max_n-n, s.device)], dim=0) if n != max_n else s  # [max_n, 3*conv_size]
                enc_sents_padded.append(padded_s)
                start_idx = end_idx

            # pad enc_sentences to max_n
            enc_sents_padded = torch.stack(enc_sents_padded, dim=0).contiguous()  # [batch, max_n, 3*conv_size]
            #print("enc_sents_padded_size")
            #print(enc_sents_padded.size())

            # pad enc_candidates to max_n
            def zero_enc_candidate(n, device):
                z = torch.zeros(n, self.num_candidates, self._candidate_sent_enc.output_size).to(device)
                return z

            # a list of [num_sents, num_cands, 3*conv_size] -> [batch, max_n, num_cands, 3*conv_size]
            enc_candidates_out = torch.stack([torch.cat([s, zero_enc_candidate(max_n - n, s.device)], dim=0)
                                              if n != max_n else s for s, n in zip(enc_candidates, sent_nums)],
                                             dim=0)  # [batch, max_n, num_cands, 3*conv_size]

        # compute context-aware embedding for each article sentence
        lstm_out = self._art_enc(enc_sents_padded, sent_nums)  # [batch, max_n, 2 * lstm_hidden]
        #print("lstm_out.size()")
        #print(lstm_out.size())
        #print("enc_candidates_out_size")
        #print(enc_candidates_out.size())

        return lstm_out, enc_candidates_out

    def _article_encode(self, article, device, pad_idx=0):
        # process one article
        # article: [num_sents * num_cands, sent_len]
        # create input for BERT
        num_sents_all, sent_len = article.size()
        #print("article_size")
        #print(article.size())

        # padding if needed
        """
        if sent_len <= 2:
            left_padding = article.new_zeros(num_sents_all, 2)
            right_padding = article.new_zeros(num_sents_all, 2)
            article = torch.cat([left_padding, article, right_padding], dim=1)
        elif sent_len <= 4:
            left_padding = article.new_zeros(num_sents_all, 1)
            right_padding = article.new_zeros(num_sents_all, 1)
            article = torch.cat([left_padding, article, right_padding], dim=1)
        """

        attention_mask = torch.ne(article, 0).to(article.device)
        token_embeddings = self._sentence_encoder.encode_word(article, attention_mask)
        # [num_sents * num_cands, sent_len, num_bert_layers * 768]

        emb_input = self._bert_w(token_embeddings)  # [num_sents * num_cands, sent_len, emb_dim]

        enc_candidate = self._candidate_sent_enc(emb_input)
        enc_candidate = enc_candidate.view(num_sents_all // self.num_candidates,self.num_candidates, -1)
        # [num_doc_sents, num_cands, 3 * conv_size]
        #print("enc_candidate_size:")
        #print(enc_candidate.size())
        return enc_candidate


class HierarchicalLSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop, auto_stop=False):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop
        self._auto_stop = auto_stop
        if auto_stop:
            self._stop = nn.Parameter(torch.Tensor(input_dim))
            init.uniform_(self._stop, -INI, INI)

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        if self._auto_stop:
            batch_size, max_sent_num, input_dim = attn_mem.size()
            # insert stop representation
            attn_mem = torch.cat([attn_mem, torch.zeros(batch_size, 1, input_dim).to(attn_mem.device)], dim=1)  # [batch, max_sent_num+1, input_dim]
            mem_sizes_tensor = torch.LongTensor(mem_sizes).to(attn_mem.device)
            attn_mem[:, mem_sizes_tensor, :] = self._stop.unsqueeze(0).expand(batch_size, -1)  # [batch_size, input_dim]
            mem_sizes = [s+1 for s in mem_sizes]

        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k, disable_selected_mask=False):
        """extract k sentences, decode only, batch_size==1"""
        # atten_mem: Tensor of size [1, max_sent_num, input_dim]
        if self._auto_stop:
            end_step = attn_mem.size(1)
            attn_mem = torch.cat([attn_mem, self._stop.view(1, 1, -1)], dim=1)  # [1, max_sent_num+1, input_dim]

        use_selected_mask = not disable_selected_mask
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        num_extracted_sent = 0
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)  # [1, 1, Ns]
            score = score.squeeze()  # [Ns]
            # set logit to -inf if the sentence is selected before
            if use_selected_mask:
                for e in extracts:
                    score[e] = -1e6
            ext = score.max(dim=0)[1].item()
            if self._auto_stop:  # break the loop if eos is selected, does not include eos to the extracts
                if ext == end_step:
                    break
            extracts.append(ext)
            num_extracted_sent += 1
            if (not self._auto_stop and num_extracted_sent == k) or num_extracted_sent == MAX_EXT:
                break
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]
        return extracts

    def _prepare(self, attn_mem):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0)
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output