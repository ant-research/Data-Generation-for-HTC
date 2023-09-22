from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .newgraph import GraphEncoder
import pdb
import nltk
import yake
from .yake_extractor import get_extractor

class BertPoolingLayer(nn.Module):
    def __init__(self, config, avg='cls'):
        super(BertPoolingLayer, self).__init__()
        self.avg = avg

    def forward(self, x):
        if self.avg == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)
        return x


class BertOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class NTXent(nn.Module):

    def __init__(self, config, tau=1.):
        super(NTXent, self).__init__()
        self.tau = tau
        self.norm = 1.
        self.transform = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, labels=None):
        x = self.transform(x)
        n = x.shape[0]
        x = F.normalize(x, p=2, dim=1) / np.sqrt(self.tau)
        # 2B * 2B
        sim = x @ x.t()
        sim[np.arange(n), np.arange(n)] = -1e9

        logprob = F.log_softmax(sim, dim=1)

        m = 2

        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)
        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm

        return loss


class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    last_hidden_state = None
    pooler_output = None
    hidden_states = None
    past_key_values = None
    attentions = None
    cross_attentions = None
    input_embeds = None


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            embedding_weight=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if embedding_weight is not None:
            if len(embedding_weight.size()) == 2:
                embedding_weight = embedding_weight.unsqueeze(-1)
            inputs_embeds = inputs_embeds * embedding_weight
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = None
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding_weight=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, inputs_embeds = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            embedding_weight=embedding_weight,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output, inputs_embeds) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            inputs_embeds=inputs_embeds,
        )


class ContrastModel(BertPreTrainedModel):
    def __init__(self, config, cls_loss=True, contrast_loss=True, graph=False, layer=1, data_path=None,
                 multi_label=False, lamb=1, threshold=0.01, tau=1, args=None):
        super(ContrastModel, self).__init__(config)
        self.config = config
        self.args = args
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel(config)
        self.pooler = BertPoolingLayer(config, 'cls')
        self.contrastive_lossfct = NTXent(config)
        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.token_classifier = BertOutputLayer(config)
        self.thre = threshold
        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau, args=self.args)
        self.lamb = lamb
        self.stopwords = set(nltk.corpus.stopwords.words())
        self.init_weights()
        self.multi_label = multi_label

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        contrast_mask = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,

        )
        pooled_output = outputs[0]
        pooled_output = self.dropout(self.pooler(pooled_output))

        loss = 0
        contrastive_loss = None
        contrast_logits = None

        logits = self.classifier(pooled_output)

        if labels is not None:
            if not self.multi_label:
                loss_fct = CrossEntropyLoss()
                target = labels.view(-1)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                target = labels.to(torch.float32)

            if self.cls_loss:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss += loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss += loss_fct(logits.view(-1, self.num_labels), target)

            if self.training:
                contrast_mask = self.graph_encoder(outputs['inputs_embeds'],
                                                   attention_mask, labels, lambda x: self.bert.embeddings(x)[0])

                contrast_output = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    embedding_weight=contrast_mask,
                )
                contrast_sequence_output = self.dropout(self.pooler(contrast_output[0]))
                contrast_logits = self.classifier(contrast_sequence_output)
                contrastive_loss = self.contrastive_lossfct(
                    torch.cat([pooled_output, contrast_sequence_output], dim=0), )

                loss += loss_fct(contrast_logits.view(-1, self.num_labels), target) 
            # import pdb
            # pdb.set_trace()
            if contrastive_loss is not None and self.contrast_loss:
                loss += contrastive_loss * self.lamb

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
        }


    def get_keywords( self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        contrast_mask = None 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,)


        contrast_mask = self.graph_encoder(outputs['inputs_embeds'],attention_mask, labels, lambda x: self.bert.embeddings(x)[0],is_training=False)
        ## contrast_mask -> torch.Tensor[batch_size,sentence_len]
        batch_size,sentence_len = contrast_mask.shape[0],contrast_mask.shape[1]
        texts = []

        for i in range(batch_size):
            l = 0
            r = 1
            # print(self.tokenizer.decode(input_ids[i]))
            while l < sentence_len and r < sentence_len:
                if input_ids[i][r].item() == 0:
                    break
                token = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                while token.startswith("##"):
                    r += 1
                    token = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                contrast_mask[i][l:r] = contrast_mask[i][l].item()
                r += 1
                l = r-1

            # tmp = [(self.tokenizer._convert_id_to_token(input_ids[i][j].item()),contrast_mask[i][j].item()) for j in range(sentence_len)]
            # pdb.set_trace()
            
            ##将每个位置的mask概率替换为这个位置的样本
        # pdb.set_trace()
        raw_texts = []
        for i in range(batch_size):
            line = ""
            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                token = self.tokenizer._convert_id_to_token(input_ids[i][j])
                if token.startswith("##"):
                    token = token[2:]
                else:
                    token = " "+token
                line += token
            raw_texts.append(line)
        
        for i in range(batch_size):
            line = ""
            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                if contrast_mask[i][j] > self.thre and input_ids[i][j] not in [0,101,102,103]:
                    token = self.tokenizer._convert_id_to_token(input_ids[i][j])
                    if token.startswith("##"):
                        token = token[2:]
                    else:
                        token = " "+token
                    line += token
            texts.append(line)
        # pdb.set_trace()
        return texts

    def get_keywords_sorted_by_fix_ratio( self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        contrast_mask = None 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,)


        contrast_mask = self.graph_encoder(outputs['inputs_embeds'],attention_mask, labels, lambda x: self.bert.embeddings(x)[0],is_training=False)


        ## contrast_mask -> torch.Tensor[batch_size,sentence_len]
        batch_size,sentence_len = contrast_mask.shape[0],contrast_mask.shape[1]
        texts = []
        token_prob_list = []
        sub_token_prob_list = []
        for i in range(batch_size):
            line = []

            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                sub_token = self.tokenizer._convert_id_to_token(input_ids[i][j].item())
                line.append((sub_token,contrast_mask[i][j]))
            sub_token_prob_list.append(line)
        
        for i in range(batch_size):
            l = 0
            r = 1
            sentence = []
            # print(self.tokenizer.decode(input_ids[i]))
            while l < sentence_len and r < sentence_len:
                if input_ids[i][r].item() == 0:
                    break
                token = ""
                prob = 0
                sub_token_b = self.tokenizer._convert_id_to_token(input_ids[i][l].item())
                token += sub_token_b
                sub_token_i  = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                while r < sentence_len and sub_token_i.startswith("##"):
                    token += sub_token_i[2:]
                    r += 1
                    if r < sentence_len:
                        sub_token_i = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                prob = torch.max(contrast_mask[i][l:r]).item()
                # import pdb
                # pdb.set_trace()
                if token not in self.stopwords and token.isalpha():
                    sentence.append((token,prob))    
                r += 1
                l = r-1
            token_prob_list.append(sentence)

                
        # pdb.set_trace()


        ratio = 0.15
        raw_texts = []
        for i in range(batch_size):
            line = ""
            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                token = self.tokenizer._convert_id_to_token(input_ids[i][j])
                if token.startswith("##"):
                    token = token[2:]
                else:
                    token = " "+token
                line += token
            raw_texts.append(line)
        
        texts = []
        thre_list = []
        for i in range(len(token_prob_list)):
            line = token_prob_list[i]
            total_num = len(line)
            sorted_line = sorted(line,key=lambda x: x[1],reverse=True)
        
            thre = min(self.thre,sorted_line[int(total_num*ratio)][1])
            thre_list.append((self.thre,sorted_line[int(total_num*ratio)][1]))
            new_line = []
            for item in line:
                if item[1] >= thre:
                    new_line.append(item[0])
            texts.append(new_line)

        #     import pdb
        #     pdb.set_trace()
        # pdb.set_trace()

        # for item in token_prob_list:

        keywords_list = []
        for line in texts:
            new_line = " ".join(line)
            keywords_list.append(new_line)
            
        return keywords_list


class ContrastModel2(BertPreTrainedModel):
    #generate mask
    def __init__(self, config, cls_loss=True, contrast_loss=True, graph=True, layer=1, data_path=None,
                 multi_label=False, lamb=1, threshold=0.01, tau=1, args=None):
        super(ContrastModel2, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bert = BertModel(config)
        self.pooler = BertPoolingLayer(config, 'cls')
        self.contrastive_lossfct = NTXent(config)
        self.cls_loss = cls_loss
        self.contrast_loss = contrast_loss
        self.token_classifier = BertOutputLayer(config)
        self.thre = threshold
        self.graph_encoder = GraphEncoder(config, graph, layer=layer, data_path=data_path, threshold=threshold, tau=tau, args=self.args)
        self.lamb = lamb
        self.stopwords = set(nltk.corpus.stopwords.words())
        self.init_weights()
        self.multi_label = multi_label
        self.extractor = get_extractor()

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_ids.device
        contrast_mask = None
        sentence_len = torch.sum(attention_mask,dim=-1)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,

        )

        # import pdb
        # pdb.set_trace()
        pooled_output = outputs[0]
        pooled_output = self.dropout(self.pooler(pooled_output))

        loss = 0
        contrastive_loss = None
        contrast_logits = None 
        contrast_logits_pos = None 
        contrast_logits_neg = None 
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            if not self.multi_label:
                loss_fct = CrossEntropyLoss()
                target = labels.view(-1)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                target_pos = labels.to(torch.float32)
                target_neg = torch.zeros(labels.shape).to(device)
            

            contrast_mask = self.graph_encoder(outputs['inputs_embeds'],
                                                attention_mask, labels, lambda x: self.bert.embeddings(x)[0])
            grad_mask = contrast_mask + (1-contrast_mask).detach()
            mask_pos = torch.zeros(contrast_mask.shape).to(device)
            mask_neg = torch.zeros(contrast_mask.shape).to(device)

            # if self.args.sigmoid:

        for i in range(contrast_mask.shape[0]):
            line = contrast_mask[i]
            length = sentence_len[i].item()
            klarge = max(int(length * 0.15),1)
            ksmall = max(int(length * 0.15),1)
            largek = torch.topk(line[:length],k=klarge)[0][-1].item()
            smallk = torch.topk(line[:length],k=ksmall,largest=False)[0][-1].item()
            line_mask_large = line > largek
            line_mask_small = line < smallk
            mask_pos[i] = line_mask_large * grad_mask[i] * attention_mask[i]
            mask_neg[i] = line_mask_small * grad_mask[i] * attention_mask[i]
            # import pdb
            # pdb.set_trace()   
            # else:
            #     temp = self.thre
            #     mask1 = contrast_mask > temp

            #     mask_pos = contrast_mask + (1 - contrast_mask).detach()
            #     mask_pos = mask_pos * mask1
            

        output_pos = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=mask_pos,
        )

        contrast_sequence_output_pos = self.dropout(self.pooler(output_pos[0]))
        contrast_logits_pos = self.classifier(contrast_sequence_output_pos)
        loss1 = loss_fct(contrast_logits_pos.view(-1, self.num_labels), target_pos)
        


        output_neg = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=mask_neg,
        ) 

        contrast_sequence_output_neg = self.dropout(self.pooler(output_neg[0]))
        contrast_logits_neg = self.classifier(contrast_sequence_output_neg)
        loss2 = loss_fct(contrast_logits_neg.view(-1, self.num_labels), target_neg)

        loss = loss1 + loss2 
        # import pdb
        # pdb.set_trace()
        # if contrastive_loss is not None and self.contrast_loss:
        #     loss += contrastive_loss * self.lamb

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        logits = contrast_logits_pos
        return {
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'contrast_logits': contrast_logits,
        }


    def get_keywords( self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        contrast_mask = None 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,)


        contrast_mask = self.graph_encoder(outputs['inputs_embeds'],attention_mask, labels, lambda x: self.bert.embeddings(x)[0],is_training=False)
        ## contrast_mask -> torch.Tensor[batch_size,sentence_len]
        batch_size,sentence_len = contrast_mask.shape[0],contrast_mask.shape[1]
        texts = []

        for i in range(batch_size):
            l = 0
            r = 1
            # print(self.tokenizer.decode(input_ids[i]))
            while l < sentence_len and r < sentence_len:
                if input_ids[i][r].item() == 0:
                    break
                token = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                while token.startswith("##"):
                    r += 1
                    token = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                contrast_mask[i][l:r] = contrast_mask[i][l].item()
                r += 1
                l = r-1

            # tmp = [(self.tokenizer._convert_id_to_token(input_ids[i][j].item()),contrast_mask[i][j].item()) for j in range(sentence_len)]
            # pdb.set_trace()
            
            ##将每个位置的mask概率替换为这个位置的样本
        # pdb.set_trace()
        raw_texts = []
        for i in range(batch_size):
            line = ""
            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                token = self.tokenizer._convert_id_to_token(input_ids[i][j])
                if token.startswith("##"):
                    token = token[2:]
                else:
                    token = " "+token
                line += token
            raw_texts.append(line)
        
        for i in range(batch_size):
            line = ""
            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                if contrast_mask[i][j] > self.thre and input_ids[i][j] not in [0,101,102,103]:
                    token = self.tokenizer._convert_id_to_token(input_ids[i][j])
                    if token.startswith("##"):
                        token = token[2:]
                    else:
                        token = " "+token
                    line += token
            texts.append(line)
        # pdb.set_trace()
        return texts

    def get_keywords_sorted_by_fix_ratio( self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,):
        contrast_mask = None 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,)


        contrast_mask = self.graph_encoder(outputs['inputs_embeds'],attention_mask, labels, lambda x: self.bert.embeddings(x)[0],is_training=False)
        # import pdb
        # pdb.set_trace()
        # import pdb
        # pdb.set_trace()
        ## contrast_mask -> torch.Tensor[batch_size,sentence_len]
        batch_size,sentence_len = contrast_mask.shape[0],contrast_mask.shape[1]
        per_sentence_len = attention_mask.sum(dim=-1)
        texts = []
        token_prob_list = []
        sub_token_prob_list = []
        for i in range(batch_size):
            line = []

            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                sub_token = self.tokenizer._convert_id_to_token(input_ids[i][j].item())
                line.append((sub_token,contrast_mask[i][j].item()))
            sub_token_prob_list.append(line)
        
        for i in range(batch_size):
            l = 0
            r = 1
            sentence = []
            # print(self.tokenizer.decode(input_ids[i]))
            while l < sentence_len and r < sentence_len:
                if input_ids[i][r].item() == 0:
                    break
                token = ""
                prob = 0
                sub_token_b = self.tokenizer._convert_id_to_token(input_ids[i][l].item())
                token += sub_token_b
                sub_token_i  = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                while r < sentence_len and sub_token_i.startswith("##"):
                    token += sub_token_i[2:]
                    r += 1
                    if r < sentence_len:
                        sub_token_i = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                prob = torch.max(contrast_mask[i][l:r]).item()
                # import pdb
                # pdb.set_trace()
                if token not in self.stopwords and token.isalpha():
                    sentence.append((token,prob))    
                r += 1
                l = r-1
            token_prob_list.append(sentence)

                


        ratio = 0.15
        raw_texts = []
        for i in range(batch_size):
            line = ""
            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                token = self.tokenizer._convert_id_to_token(input_ids[i][j])
                if token.startswith("##"):
                    token = token[2:]
                else:
                    token = " "+token
                line += token
            raw_texts.append(line)
        
        texts = []
        thre_list = []
        for i in range(len(token_prob_list)):
            line = token_prob_list[i]
            total_num = len(line)
            sorted_line = sorted(line,key=lambda x: x[1],reverse=True)
        
            thre = sorted_line[min(min(int(total_num*r),15),total_num-1)][1]
            # thre = sorted_line[int(total_num*ratio)][1]
            thre_list.append((self.thre,sorted_line[int(total_num*ratio)][1]))
            new_line = []
            for item in line:
                if item[1] >= thre:
                    new_line.append(item[0])
            texts.append(new_line)

        # import pdb
        # pdb.set_trace()
        # pdb.set_trace()

        # for item in token_prob_list:
        # import pdb
        # pdb.set_trace()
        keywords_list = []
        for line in texts:
            new_line = " ".join(line)
            keywords_list.append(new_line)
            
        return keywords_list


    def get_token_prob_list( self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,):
        contrast_mask = None 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,)


        contrast_mask = self.graph_encoder(outputs['inputs_embeds'],attention_mask, labels, lambda x: self.bert.embeddings(x)[0],is_training=False)
        # import pdb
        # pdb.set_trace()
        # import pdb
        # pdb.set_trace()
        ## contrast_mask -> torch.Tensor[batch_size,sentence_len]
        batch_size,sentence_len = contrast_mask.shape[0],contrast_mask.shape[1]
        per_sentence_len = attention_mask.sum(dim=-1)
        texts = []
        token_prob_list = []
        sub_token_prob_list = []
        for i in range(batch_size):
            line = []

            for j in range(sentence_len):
                if input_ids[i][j].item() == 0:
                    break
                sub_token = self.tokenizer._convert_id_to_token(input_ids[i][j].item())
                line.append((sub_token,contrast_mask[i][j].item()))
            sub_token_prob_list.append(line)
        
        for i in range(batch_size):
            l = 0
            r = 1
            sentence = []
            # print(self.tokenizer.decode(input_ids[i]))
            while l < sentence_len and r < sentence_len:
                if input_ids[i][r].item() == 0:
                    break
                token = ""
                prob = 0
                sub_token_b = self.tokenizer._convert_id_to_token(input_ids[i][l].item())
                token += sub_token_b
                sub_token_i  = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                while r < sentence_len and sub_token_i.startswith("##"):
                    token += sub_token_i[2:]
                    r += 1
                    if r < sentence_len:
                        sub_token_i = self.tokenizer._convert_id_to_token(input_ids[i][r].item())
                prob = torch.max(contrast_mask[i][l:r]).item()
                # import pdb
                # pdb.set_trace()
                if token not in self.stopwords and token.isalpha():
                    sentence.append((token,prob))    
                r += 1
                l = r-1
            
            token_prob_list.append(sentence)
                
        return token_prob_list

    def select_keywords(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ori_texts=None,):
        contrast_mask = None 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            embedding_weight=contrast_mask,)


        contrast_mask = self.graph_encoder(outputs['inputs_embeds'],attention_mask, labels, lambda x: self.bert.embeddings(x)[0],is_training=False)

        batch_size,sentence_len = contrast_mask.shape[0],contrast_mask.shape[1]


        keywords_list = []

        for i in range(batch_size):
            text = ori_texts[i].split("\t")[0]
            keywords = self.extractor.extract_keywords(text)
            new_keywords = []
            for keyword in keywords:

                id_list = self.tokenizer.encode(keyword[0].strip().lower())[1:-1]

                pos = self.search_pos(input_ids[i],id_list)
                # if pos == -1:
                #     import pdb
                #     pdb.set_trace()
                # assert pos != -1, "没找到关键词在原文本中的位置"

                if pos != -1:
                    score = torch.max(contrast_mask[i][pos:pos+len(id_list)])
                    new_keywords.append((keyword[0],keyword[1],score.item()))
            new_keywords = sorted(new_keywords,key=lambda item : item[2],reverse=True)
            keywords_list.append([item[0] for item in new_keywords[:20]])
        
        result = []
        for i in range(batch_size):
            result.append(','.join(keywords_list[i]))
        

        return result
    
    def search_pos(self,input_ids,keyword_ids):
        '''
            input:
                input_ids -> Tensor[512]
                keyword_ids -> List[keyword_len]
            output:
                -1 : 没找到
                其他 : 匹配的第一个位置
        '''
        # import pdb
        # pdb.set_trace()
        for i in range(input_ids.shape[0]):
            flag = 1
            for j in range(len(keyword_ids)):
                if input_ids[i+j].item() != keyword_ids[j]:
                    flag = -1
                    break
            if flag == 1:
                return i
        return -1        
