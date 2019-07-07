from typing import List

import torch
import torch.nn as nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pyknp import Juman


class BertPosNegClassifier(nn.Module):
    def __init__(self,
                 bert_model: str,
                 ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self,
                input_ids: torch.Tensor,  # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ) -> torch.Tensor:  # (b, label)
        # (b, hid)
        _, pooled_output = self.bert(input_ids,
                                     attention_mask=attention_mask,
                                     output_all_encoded_layers=False)
        logits = self.classifier(self.dropout(pooled_output))
        return logits


class SentimentAnalysis:
    def __init__(self, bert_model: str, fine_tuned_model: str, jumanpp_command: str):
        self.jumanpp = Juman(command=jumanpp_command)

        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
        self.model = BertPosNegClassifier(bert_model)
        state_dict = torch.load(fine_tuned_model, map_location=torch.device('cpu'))
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        self.model.eval()

    def get_prediction(self, sentence: str) -> int:
        print(sentence)
        text: str = self._segmentation(sentence)

        tokenized_text: List[str] = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']

        indexed_tokens: List[int] = [self.tokenizer.convert_tokens_to_ids(tokenized_text)]

        tokens_tensor = torch.tensor(indexed_tokens)
        attention_mask_tensor = torch.tensor([[1] * len(tokenized_text)])
        # segments_tensors = torch.tensor([ [0] * len(indexed_tokens_list[0]) for _ in range(text_length)])

        output: torch.Tensor = self.model(tokens_tensor, attention_mask=attention_mask_tensor)
        prediction: int = torch.argmax(output[0]).item()  # 0 or 1

        if prediction == 0:
            prediction = -1

        return prediction

    def _segmentation(self, text: str) -> str:
        result = self.jumanpp.analysis(text)
        return ' '.join(mrph.midasi for mrph in result.mrph_list())
