import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

# bert classifier split into embedding and classification head
# substituting tanh in pooler with ReLU
class CustomBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    #non -negative embedding pooler uses relu instesad of tanh
    def features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return torch.relu(self.bert.pooler.dense(cls_token))
    #classification head for the activations
    def end_model(self, activations: torch.Tensor):
        return self.classifier(activations)
    # combination of both 
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.end_model(self.features(input_ids, attention_mask))
