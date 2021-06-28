import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .attention import step_attention
from .util import sequence_mean, len_mask
from .copy_summ import CopySumm, CopyLSTMDecoder
from . import beam_search as bs
from .rnn import lstm_encoder


class CopyCondSumm(CopySumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout=0.0):
        super().__init__(vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout)
        # multiplicative attention
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._attn_wm_external = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq_external = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_wm_external)
        init.xavier_normal_(self._attn_wq_external)

        self._mem_enc_lstm = nn.LSTM(emb_dim, n_hidden, n_layer, bidirectional=bidirectional, dropout=dropout)
        # project decoder output to emb_dim, then
        # apply weight matrix from embedding layer
        self._projection = nn.Sequential(
            nn.Linear(3 * n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )
        self._decoder = CopyCondLSTMDecoder(self._attn_wq_external, self._copy, self._embedding, self._dec_lstm,
                                            self._attn_wq, self._projection)

    def encode(self, article, raw_memory, art_lens=None, memory_lens=None):  # article: [batch_size, seq_len]
        # encode document
        size = (
            self._init_enc_h.size(0),
            len(art_lens) if art_lens else 1,
            self._init_enc_h.size(1)
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        enc_art, final_states = lstm_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding
        )  # enc_art: [seq_len, batch, 2*hidden_dim], final_states: ([1, batch, 2*hidden_dim], [1, batch, 2*hidden_dim])
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)  # [1, batch_size, hidden_dim]
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)  # [batch, seq_len, hidden_dim]
        init_attn_out = sequence_mean(attention, art_lens, dim=1)  # [batch, hidden_dim]

        # encode memory
        enc_mem, _ = lstm_encoder(raw_memory, self._mem_enc_lstm, memory_lens, init_states=None,
                                             embedding=self._embedding)  # enc_mem: [seq_len, batch, 2*hidden_dim]
        attention_external = torch.matmul(enc_mem, self._attn_wm_external).transpose(0, 1)  # [batch, seq_len, hidden_dim]

        init_ext_attn_out = sequence_mean(attention_external, memory_lens, dim=1)  # [batch, hidden_dim]

        init_dec_out = self._projection(torch.cat(
            [init_h[-1], init_attn_out, init_ext_attn_out], dim=1
        ))  # [batch, 3 * hidden_dim] -> [batch, embed_size]

        return attention, attention_external, (init_dec_states, init_dec_out)

    def forward(self, article, art_lens, memory, mem_lens, abstract, extend_art, extend_vsize):
        attention, attention_external, init_dec_states = self.encode(article, memory, art_lens, mem_lens)
        #encoded_memory, init_memory_out = self.encode_external_memory(memory, mem_lens)
        art_mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        mem_mask = len_mask(mem_lens, memory.device).unsqueeze(-2)

        logit = self._decoder(
            (attention, art_mask, attention_external, mem_mask, extend_art, extend_vsize),
            init_dec_states, abstract
        )
        return logit

    def batch_decode(self, article, art_lens, memory, mem_lens, extend_art, extend_vsize,
                     go, eos, unk, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, attention_external, init_dec_states = self.encode(article, memory, art_lens, mem_lens)
        art_mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        mem_mask = len_mask(mem_lens, memory.device).unsqueeze(-2)
        attention = (attention, art_mask, attention_external, mem_mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go] * batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            attns.append(attn_score)
            outputs.append(tok[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
        return outputs, attns

    def decode(self, article, memory, extend_art, extend_vsize, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        attention, attention_external, init_dec_states = self.encode(article, memory)
        attention = (attention, None, attention_external, None, extend_art, extend_vsize)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns

    def batched_beamsearch(self, article, art_lens, memory, mem_lens,
                           extend_art, extend_vsize,
                           go, eos, unk, max_len, beam_size, diverse=1.0):
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, attention_external, init_dec_states = self.encode(article, memory, art_lens, mem_lens)
        art_mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        mem_mask = len_mask(mem_lens, memory.device).unsqueeze(-2)
        all_attention = (attention, art_mask, attention_external, mem_mask, extend_art, extend_vsize)
        attention = all_attention
        (h, c), prev = init_dec_states
        all_beams = [bs.init_beam(go, (h[:, i, :], c[:, i, :], prev[i]))
                     for i in range(batch_size)]
        finished_beams = [[] for _ in range(batch_size)]
        outputs = [None for _ in range(batch_size)]
        for t in range(max_len):
            toks = []
            all_states = []
            for beam in filter(bool, all_beams):
                token, states = bs.pack_beam(beam, article.device)
                toks.append(token)
                all_states.append(states)
            token = torch.stack(toks, dim=1)
            states = ((torch.stack([h for (h, _), _ in all_states], dim=2),
                       torch.stack([c for (_, c), _ in all_states], dim=2)),
                      torch.stack([prev for _, prev in all_states], dim=1))
            token.masked_fill_(token >= vsize, unk)

            topk, lp, states, attn_score = self._decoder.topk_step(
                token, states, attention, beam_size)

            batch_i = 0
            for i, (beam, finished) in enumerate(zip(all_beams,
                                                     finished_beams)):
                if not beam:
                    continue
                finished, new_beam = bs.next_search_beam(
                    beam, beam_size, finished, eos,
                    topk[:, batch_i, :], lp[:, batch_i, :],
                    (states[0][0][:, :, batch_i, :],
                     states[0][1][:, :, batch_i, :],
                     states[1][:, batch_i, :]),
                    attn_score[:, batch_i, :],
                    diverse
                )
                batch_i += 1
                if len(finished) >= beam_size:
                    all_beams[i] = []
                    outputs[i] = finished[:beam_size]
                    # exclude finished inputs
                    (attention, art_mask, attention_external, mem_mask, extend_art, extend_vsize
                    ) = all_attention
                    art_masks = [art_mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    mem_masks = [mem_mask[j] for j, o in enumerate(outputs)
                             if o is None]
                    ind = [j for j, o in enumerate(outputs) if o is None]
                    ind = torch.LongTensor(ind).to(attention.device)
                    attention, attention_external, extend_art = map(
                        lambda v: v.index_select(dim=0, index=ind),
                        [attention, attention_external, extend_art]
                    )
                    if art_masks:
                        art_mask = torch.stack(art_masks, dim=0)
                        mem_mask = torch.stack(mem_masks, dim=0)
                    else:
                        art_mask = None
                        mem_mask = None
                    attention = (
                        attention, art_mask, attention_external, mem_mask, extend_art, extend_vsize)
                else:
                    all_beams[i] = new_beam
                    finished_beams[i] = finished
            if all(outputs):
                break
        else:
            for i, (o, f, b) in enumerate(zip(outputs,
                                              finished_beams, all_beams)):
                if o is None:
                    outputs[i] = (f+b)[:beam_size]
        return outputs


class CopyCondLSTMDecoder(CopyLSTMDecoder):
    def __init__(self, attn_w_external, copy, *args, **kwargs):
        self._attn_w_external = attn_w_external
        super().__init__(copy, *args, **kwargs)

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)  # [batch_size, hidden_dim]
        query_external = torch.mm(lstm_out, self._attn_w_external)  # [batch_size, hidden_dim]
        attention, attn_mask, attention_external, attn_mask_external, extend_src, extend_vsize = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        context_external, score_external = step_attention(query_external, attention_external, attention_external, attn_mask_external)

        dec_out = self._projection(torch.cat([lstm_out, context, context_external], dim=1))  # [batch, 3*hidden_dim]

        # extend generation prob to extended vocabulary
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)
        # compute the probabilty of each copying
        copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in))
        # add the copy prob to existing vocab distribution
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score),
                source=score * copy_prob
        ) + 1e-8)  # numerical stability for log
        return lp, (states, dec_out), score

    def topk_step(self, tok, states, attention, k):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        (h, c), prev_out = states

        # lstm is not bemable
        nl, _, _, d = h.size()
        beam, batch = tok.size()
        lstm_in_beamable = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)
        lstm_in = lstm_in_beamable.contiguous().view(beam*batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]

        # attention is beamable
        query = torch.matmul(lstm_out, self._attn_w)
        query_external = torch.matmul(lstm_out, self._attn_w_external)
        attention, attn_mask, attention_external, attn_mask_external, extend_src, extend_vsize = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        context_external, score_external = step_attention(query_external, attention_external, attention_external,
                                                          attn_mask_external)
        dec_out = self._projection(torch.cat([lstm_out, context, context_external], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch*beam, -1), extend_vsize)
        copy_prob = torch.sigmoid(
            self._copy(context, lstm_out, lstm_in_beamable)
        ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score).contiguous().view(
                    beam*batch, -1),
                source=score.contiguous().view(beam*batch, -1) * copy_prob
        ) + 1e-8).contiguous().view(beam, batch, -1)

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, (states, dec_out), score
