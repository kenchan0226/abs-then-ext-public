import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .extract import LSTMPointerNet


INI = 1e-2
MAX_EXT = 7


class PtrExtractorRL(nn.Module):
    """ works only on single sample in RL setting"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            if self.training:
                prob = F.softmax(score, dim=-1)
                out = torch.distributions.Categorical(prob)
            else:
                for o in outputs:
                    score[0, o[0, 0].item()][0] = -1e18
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            lstm_in = attn_mem[out[0, 0].item()].unsqueeze(0)
            lstm_states = (h, c)
        return outputs

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention + torch.mm(query, w)
        score = torch.mm(F.tanh(sum_), v.unsqueeze(1)).t()
        return score

    @staticmethod
    def attention(attention, query, v, w):
        """ attention context vector"""
        score = F.softmax(
            PtrExtractorRL.attention_score(attention, query, v, w), dim=-1)
        output = torch.mm(score, attention)
        return output


class PtrExtractorRLStop(PtrExtractorRL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if args:
            ptr_net = args[0]
        else:
            ptr_net = kwargs['ptr_net']
        assert isinstance(ptr_net, LSTMPointerNet)
        self._stop = nn.Parameter(
            torch.Tensor(self._lstm_cell.input_size))
        init.uniform_(self._stop, -INI, INI)

    def forward(self, attn_mem, n_ext=None, disable_selected_mask=False):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        if n_ext is not None:
            return super().forward(attn_mem, n_ext)
        use_selected_mask = not disable_selected_mask
        max_step = attn_mem.size(0)
        attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            if use_selected_mask:
                for o in outputs:
                    score[0, o.item()] = -1e18
            #if len(outputs) >= MAX_EXT:
            #    score[0, max_step] = 1e10
            if self.training:
                prob = F.softmax(score, dim=-1)
                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            if out.item() == max_step:
                break
            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = (h, c)
        if dists:
            # return distributions only when not empty (trining)
            return outputs, dists
        else:
            return outputs


class PtrExtractorWithCandidateRLStop(PtrExtractorRLStop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, attn_mem, n_ext=None, num_candidates=1, disable_selected_mask=False):
        """atten_mem: Tensor of size [num_sents * num_cands, input_dim]"""
        if n_ext is not None:
            return super().forward(attn_mem, n_ext)
        use_selected_mask = not disable_selected_mask
        max_step = attn_mem.size(0)
        attn_mem = torch.cat([attn_mem, self._stop.unsqueeze(0)], dim=0)
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        outputs = []
        masked_candidates = []
        dists = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        while True:
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrExtractorRL.attention(hop_feat, query,
                                                 self._hop_v, self._hop_wq)
            score = PtrExtractorRL.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            for e in masked_candidates:
                score[0, e] = -1e6
            #for o in outputs:
            #    score[0, o.item()] = -1e18
            if self.training:
                prob = F.softmax(score, dim=-1)
                m = torch.distributions.Categorical(prob)
                dists.append(m)
                out = m.sample()
            else:
                out = score.max(dim=1, keepdim=True)[1]
            outputs.append(out)
            if out.item() == max_step:
                break
            # map the extracted candidate idx to all the candidate indices in that sentence and mask them out
            selected_sent_idx = out.item() // num_candidates
            if use_selected_mask:
                masked_candidates += [selected_sent_idx * num_candidates + i for i in range(num_candidates)]

            lstm_in = attn_mem[out.item()].unsqueeze(0)
            lstm_states = (h, c)
        if dists:
            # return distributions only when not empty (trining)
            return outputs, dists
        else:
            return outputs

class PtrScorer(nn.Module):
    """ to be used as critic (predicts a scalar baseline reward)"""
    def __init__(self, ptr_net):
        super().__init__()
        assert isinstance(ptr_net, LSTMPointerNet)
        self._init_h = nn.Parameter(ptr_net._init_h.clone())
        self._init_c = nn.Parameter(ptr_net._init_c.clone())
        self._init_i = nn.Parameter(ptr_net._init_i.clone())
        self._lstm_cell = MultiLayerLSTMCells.convert(ptr_net._lstm)

        # attention parameters
        self._attn_wm = nn.Parameter(ptr_net._attn_wm.clone())
        self._attn_wq = nn.Parameter(ptr_net._attn_wq.clone())
        self._attn_v = nn.Parameter(ptr_net._attn_v.clone())

        # hop parameters
        self._hop_wm = nn.Parameter(ptr_net._hop_wm.clone())
        self._hop_wq = nn.Parameter(ptr_net._hop_wq.clone())
        self._hop_v = nn.Parameter(ptr_net._hop_v.clone())
        self._n_hop = ptr_net._n_hop

        # regression layer
        self._score_linear = nn.Linear(self._lstm_cell.input_size, 1)

    def forward(self, attn_mem, n_step):
        """atten_mem: Tensor of size [num_sents, input_dim]"""
        attn_feat = torch.mm(attn_mem, self._attn_wm)
        hop_feat = torch.mm(attn_mem, self._hop_wm)
        scores = []
        lstm_in = self._init_i.unsqueeze(0)
        lstm_states = (self._init_h.unsqueeze(1), self._init_c.unsqueeze(1))
        for _ in range(n_step):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[:, -1, :]
            for _ in range(self._n_hop):
                query = PtrScorer.attention(hop_feat, hop_feat, query,
                                            self._hop_v, self._hop_wq)
            output = PtrScorer.attention(
                attn_mem, attn_feat, query, self._attn_v, self._attn_wq)
            score = self._score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores

    @staticmethod
    def attention(attention, attention_feat, query, v, w):
        """ attention context vector"""
        sum_ = attention_feat + torch.mm(query, w)
        score = F.softmax(torch.mm(F.tanh(sum_), v.unsqueeze(1)).t(), dim=-1)
        output = torch.mm(score, attention)
        return output


class ActorCritic(nn.Module):
    """ shared encoder between actor/critic"""
    def __init__(self, sent_encoder, art_encoder,
                 extractor, art_batcher, num_candidates=1):
        super().__init__()
        self._sent_enc = sent_encoder
        self._art_enc = art_encoder
        self._ext = PtrExtractorRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher
        self._num_candidates = num_candidates

    def forward(self, raw_article_sents, disable_selected_mask, n_abs=None, debug=False):
        # raw_article_sents: sents in one article
        # encode
        article_sent = self._batcher(raw_article_sents)
        enc_sent = self._sent_enc(article_sent).unsqueeze(0)
        enc_art = self._art_enc(enc_sent).squeeze(0)
        # extract
        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_art)
        else:
            outputs = self._ext(enc_art, n_abs, disable_selected_mask=disable_selected_mask)
        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_art, n_abs)
            return outputs, scores
        else:
            return outputs

class ActorCriticCand(nn.Module):
    def __init__(self, candidate_sent_encoder, candidate_aggregator, art_encoder, extractor, art_batcher, num_candidates):
        super().__init__()
        self._candidate_sent_enc = candidate_sent_encoder
        self._candidate_agg = candidate_aggregator
        self._art_enc = art_encoder
        self._ext = PtrExtractorWithCandidateRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher
        self.num_candidates = num_candidates

    def forward(self, raw_article_sents, disable_selected_mask, n_abs=None, debug=False):
        # raw_article_sents: sents in one article
        # encode
        article_sents = self._batcher(raw_article_sents)
        #print("article sents: {}".format(article_sents))
        enc_candidates = self._candidate_sent_enc(article_sents).view(
            article_sents.size(0) // self.num_candidates, self.num_candidates, -1)  # [num_sents, num_cands, 3*conv_size]
        enc_sents = self._candidate_agg(enc_candidates).unsqueeze(0)  # [1, num_sents, 3*conv_size]
        lstm_out = self._art_enc(enc_sents).squeeze(0)  # [num_sents, 2 * lstm_hidden]
        # Concat local and global representation
        num_sents, lstm_out_dim = lstm_out.size()
        # new dim=[num_sents, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([enc_candidates,
                             lstm_out.unsqueeze(1).expand(num_sents, self.num_candidates,
                                                          lstm_out_dim)], dim=2)
        enc_out = enc_out.view(num_sents * self.num_candidates, -1)  # [num_sents * num_cands, 2*lstm_hidden + 3*conv_size]
        #lstm_out = self._art_enc(enc_sents)  # [1, num_sents, 2 * lstm_hidden]
        #enc_candidates = enc_candidates.unsqueeze(0)  # [1, num_sents, num_cands, 3*conv_size]
        # Concat local and global representation
        #batch_size, num_sents, lstm_out_dim = lstm_out.size()
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        #enc_out = torch.cat([enc_candidates,
        #                     lstm_out.unsqueeze(2).expand(batch_size, num_sents, self.num_candidates,
        #                                                                  lstm_out_dim)], dim=3)
        #enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)

        # extract
        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_out, num_candidates=self.num_candidates, disable_selected_mask=disable_selected_mask)
        else:
            outputs = self._ext(enc_out, n_abs, num_candidates=self.num_candidates, disable_selected_mask=disable_selected_mask)

        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_out, n_abs)
            return outputs, scores
        else:
            return outputs

class ActorCriticSentBertCand(nn.Module):
    def __init__(self, candidate_sent_encoder, bert_linear, candidate_aggregator, art_encoder, extractor, art_batcher, num_candidates):
        super().__init__()
        self._sentence_encoder = candidate_sent_encoder
        self._sentence_encoder.eval()
        for p in self._sentence_encoder.parameters():
            p.requires_grad = False
        self._bert_w = bert_linear
        self._candidate_agg = candidate_aggregator
        self._art_enc = art_encoder
        self._ext = PtrExtractorWithCandidateRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher
        self.num_candidates = num_candidates

    def forward(self, raw_article_sents, disable_selected_mask, n_abs=None, debug=False):
        # raw_article_sents: sents in one article
        # encode
        article_sents = self._batcher(raw_article_sents)
        #print("article sents: {}".format(article_sents))

        attention_mask_tensor = torch.ne(article_sents, 0).to(article_sents.device)
        sent_bert_embeddings = self._sentence_encoder.encode_tensor(article_sents, attention_mask_tensor)
        # sent_bert_embeddings: [num_sent * num_candidates, 768]
        enc_candidates = self._bert_w(sent_bert_embeddings)  # [num_sent * num_candidates, 3 * conv_size]
        enc_candidates = enc_candidates.view(article_sents.size(0) // self.num_candidates, self.num_candidates, -1)  # [num_sents, num_cands, 3*conv_size]

        enc_sents = self._candidate_agg(enc_candidates).unsqueeze(0)  # [1, num_sents, 3*conv_size]
        lstm_out = self._art_enc(enc_sents).squeeze(0)  # [num_sents, 2 * lstm_hidden]
        # Concat local and global representation
        num_sents, lstm_out_dim = lstm_out.size()
        # new dim=[num_sents, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([enc_candidates,
                             lstm_out.unsqueeze(1).expand(num_sents, self.num_candidates,
                                                          lstm_out_dim)], dim=2)
        enc_out = enc_out.view(num_sents * self.num_candidates, -1)  # [num_sents * num_cands, 2*lstm_hidden + 3*conv_size]
        #lstm_out = self._art_enc(enc_sents)  # [1, num_sents, 2 * lstm_hidden]
        #enc_candidates = enc_candidates.unsqueeze(0)  # [1, num_sents, num_cands, 3*conv_size]
        # Concat local and global representation
        #batch_size, num_sents, lstm_out_dim = lstm_out.size()
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        #enc_out = torch.cat([enc_candidates,
        #                     lstm_out.unsqueeze(2).expand(batch_size, num_sents, self.num_candidates,
        #                                                                  lstm_out_dim)], dim=3)
        #enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)

        # extract
        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_out, num_candidates=self.num_candidates, disable_selected_mask=disable_selected_mask)
        else:
            outputs = self._ext(enc_out, n_abs, num_candidates=self.num_candidates, disable_selected_mask=disable_selected_mask)

        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_out, n_abs)
            return outputs, scores
        else:
            return outputs


class ActorCriticSentWordBertCand(nn.Module):
    def __init__(self, sentence_encoder, bert_linear, candidate_sent_encoder, candidate_aggregator, art_encoder, extractor, art_batcher, num_candidates):
        super().__init__()

        self._sentence_encoder = sentence_encoder
        self._sentence_encoder.eval()
        for p in self._sentence_encoder.parameters():
            p.requires_grad = False
        self._bert_w = bert_linear
        self._candidate_sent_enc = candidate_sent_encoder
        self._candidate_agg = candidate_aggregator
        self._art_enc = art_encoder
        self._ext = PtrExtractorWithCandidateRLStop(extractor)
        self._scr = PtrScorer(extractor)
        self._batcher = art_batcher
        self.num_candidates = num_candidates

    def forward(self, raw_article_sents, disable_selected_mask, n_abs=None, debug=False):
        # raw_article_sents: sents in one article
        # encode
        article_sents = self._batcher(raw_article_sents)
        num_sents_all, sent_len = article_sents.size()

        attention_mask = torch.ne(article_sents, 0).to(article_sents.device)
        token_embeddings = self._sentence_encoder.encode_word(article_sents, attention_mask)
        # [num_sents * num_cands, sent_len, num_bert_layers * 768]

        emb_input = self._bert_w(token_embeddings)  # [num_sents * num_cands, sent_len, emb_dim]

        enc_candidates = self._candidate_sent_enc(emb_input).view(
            num_sents_all // self.num_candidates, self.num_candidates, -1)  # [num_sents, num_cands, 3*conv_size]
        enc_sents = self._candidate_agg(enc_candidates).unsqueeze(0)  # [1, num_sents, 3*conv_size]
        lstm_out = self._art_enc(enc_sents).squeeze(0)  # [num_sents, 2 * lstm_hidden]
        # Concat local and global representation
        num_sents, lstm_out_dim = lstm_out.size()
        # new dim=[num_sents, num_cand, 2*lstm_hidden + 3*conv_size]
        enc_out = torch.cat([enc_candidates,
                             lstm_out.unsqueeze(1).expand(num_sents, self.num_candidates,
                                                          lstm_out_dim)], dim=2)
        enc_out = enc_out.view(num_sents * self.num_candidates, -1)  # [num_sents * num_cands, 2*lstm_hidden + 3*conv_size]
        #lstm_out = self._art_enc(enc_sents)  # [1, num_sents, 2 * lstm_hidden]
        #enc_candidates = enc_candidates.unsqueeze(0)  # [1, num_sents, num_cands, 3*conv_size]
        # Concat local and global representation
        #batch_size, num_sents, lstm_out_dim = lstm_out.size()
        # new dim=[batch, max_n, num_cand, 2*lstm_hidden + 3*conv_size]
        #enc_out = torch.cat([enc_candidates,
        #                     lstm_out.unsqueeze(2).expand(batch_size, num_sents, self.num_candidates,
        #                                                                  lstm_out_dim)], dim=3)
        #enc_out = enc_out.view(batch_size, max_n * self.num_candidates, -1)

        # extract
        if n_abs is not None and not self.training:
            n_abs = min(len(raw_article_sents), n_abs)
        if n_abs is None:
            outputs = self._ext(enc_out, num_candidates=self.num_candidates, disable_selected_mask=disable_selected_mask)
        else:
            outputs = self._ext(enc_out, n_abs, num_candidates=self.num_candidates, disable_selected_mask=disable_selected_mask)

        if self.training:
            if n_abs is None:
                n_abs = len(outputs[0])
            scores = self._scr(enc_out, n_abs)
            return outputs, scores
        else:
            return outputs
