import torch.nn as nn
from transformers import XLMRobertaModel

class QuestionClassifier(nn.Module):
    def __init__(self, pretrained_model_name):
        super(QuestionClassifier, self).__init__()
        self.encoder_model = XLMRobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.encoder_model.pooler.dense.weight.shape[0], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

    def forward(self, input_ids, attention_mask, target=None):
        encoder_output = self.encoder_model(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = encoder_output.pooler_output
        dropout_output = self.dropout(pooler_output)
        fc1_ouput = self.fc1(dropout_output)
        proba = self.sigmoid(fc1_ouput).flatten()
        loss = None
        if target is not None:
            loss = self.loss_func(proba, target)
        return proba, loss
