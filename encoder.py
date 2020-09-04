import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel


class BiEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']

    def forward(self, context_input_ids, context_input_masks,
                            responses_input_ids, responses_input_masks, labels=None):
        ## only select the first response (whose lbl==1)
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

        context_vec = self.bert(context_input_ids, context_input_masks)[0][:,0,:]  # [bs,dim]

        batch_size, res_cnt, seq_length = responses_input_ids.shape
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)

        responses_vec = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:]  # [bs,dim]
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)

        if labels is not None:
            responses_vec = responses_vec.squeeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs]
            mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            context_vec = context_vec.unsqueeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.permute(0, 2, 1)).squeeze()
            return dot_product


class CrossEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, text_input_ids, text_input_masks, text_input_segments, labels=None):
        batch_size, neg, dim = text_input_ids.shape
        text_input_ids = text_input_ids.reshape(-1, dim)
        text_input_masks = text_input_masks.reshape(-1, dim)
        text_input_segments = text_input_segments.reshape(-1, dim)
        text_vec = self.bert(text_input_ids, text_input_masks, text_input_segments)[0][:,0,:]  # [bs,dim]
        score = self.linear(text_vec)
        score = score.view(-1, neg)
        if labels is not None:
            loss = -F.log_softmax(score, -1)[:,0].mean()
            return loss
        else:
            return score


class PolyEncoder(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = kwargs['bert']
        self.poly_m = kwargs['poly_m']
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
        # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L355
        torch.nn.init.normal_(self.poly_code_embeddings.weight, config.hidden_size ** -0.5)

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, context_input_ids, context_input_masks,
                            responses_input_ids, responses_input_masks, labels=None):
        # during training, only select the first response
        # we are using other instances in a batch as negative examples
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape # res_cnt is 1 during training

        # context encoder
        ctx_out = self.bert(context_input_ids, context_input_masks)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        # response encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        cand_emb = self.bert(responses_input_ids, responses_input_masks)[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        # merge
        if labels is not None:
            # we are recycling responses for faster training
            # we repeat responses for batch_size times to simulate test phase
            # so that every context is paired with batch_size responses
            cand_emb = cand_emb.permute(1, 0, 2) # [1, bs, dim]
            cand_emb = cand_emb.expand(batch_size, batch_size, cand_emb.shape[2]) # [bs, bs, dim]
            ctx_emb = self.dot_attention(cand_emb, embs, embs).squeeze() # [bs, bs, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1) # [bs, bs]
            mask = torch.eye(batch_size).to(context_input_ids.device) # [bs, bs]
            loss = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()
            return loss
        else:
            ctx_emb = self.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
            return dot_product
