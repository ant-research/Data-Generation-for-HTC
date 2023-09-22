from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder
from transformers.file_utils import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .graph import GraphEncoder
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel,RobertaModel




class RobertaContrastModel(RobertaPreTrainedModel):
    def __init__(self, config, cls_loss=True, contrast_loss=True, graph=False, layer=1, data_path=None,
                 multi_label=False, lamb=1, threshold=0.01, tau=1):
        super(RobertaContrastModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.roberta = RobertaModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.cls_loss = cls_loss

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


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[0]
        pooled_output = torch.tanh(self.dense(self.dropout(pooled_output[:,0,:])))


        loss = 0
  

        logits = self.classifier(self.dropout(pooled_output))

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



        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
