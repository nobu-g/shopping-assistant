import sys
import argparse
import json
import copy

import torch

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM

import xmlrpc.server as xmlrpc_server
from pyknp import Juman


class MaskedLM:
    def __init__(self, bert_model, jumanpp_command, is_char_base=False, topk=5):
        self.is_char_base = is_char_base
        self.topk = topk
        self.jumanpp = Juman(command=jumanpp_command)
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(bert_model)
        self.model.eval()

    def segmentation(self, text):
        if self.is_char_base is True:
            text = " ".join(list(text))            
        else:
            result = self.jumanpp.analysis(text)
            text = " ".join([ mrph.midasi for mrph in result.mrph_list() ])

        return text

    def get_predictions(self, sentence):
        print("{}".format(sentence))
        text = self.segmentation(sentence)

        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_text.insert(0, "[CLS]")
        tokenized_text.append("[SEP]")

        indexed_tokens_list = []
        text_length = len(tokenized_text)
        for i, token in enumerate(tokenized_text):
            new_tokenized_text = copy.deepcopy(tokenized_text)
            new_tokenized_text[i] = '[MASK]'

            indexed_tokens_list.append(self.tokenizer.convert_tokens_to_ids(new_tokenized_text))

        tokens_tensor = torch.tensor(indexed_tokens_list)
        segments_tensors = torch.tensor([ [0] * len(indexed_tokens_list[0]) for _ in range(text_length)])

        predictions = self.model(tokens_tensor, segments_tensors)
        _, indices = torch.sort(predictions, descending=True)

        prediction_results = []
        for i, token in enumerate(tokenized_text):
            if i == 0 or i == text_length - 1:
                continue

            prediction_string = self.get_prediction_string(indices[i, i, :self.topk].numpy(), token)
            prediction_results.append({ "input": token, "predictions": prediction_string })

        return json.dumps(prediction_results, ensure_ascii=False)

    def get_prediction_string(self, topk_predictions, token):
        strings = []
        for predicted_token in self.tokenizer.convert_ids_to_tokens(topk_predictions):
            if token == predicted_token:
                strings.append("<font color='red'>{}</font>".format(predicted_token))
            else:
                strings.append(predicted_token)

        return ", ".join(strings)


def main(args):
    server = xmlrpc_server.SimpleXMLRPCServer((args.server, args.port), allow_none=True)

    masked_lm = MaskedLM(args.bert_model, args.jumanpp_command)
    server.register_function(masked_lm.get_predictions, "get_predictions")
    print("loading done.", file=sys.stderr)
    server.serve_forever()


# usage: python server.py --bert_model /larch/shibata/bert/preprocess/181213_subword_wikipedia/pretraining_model_20e --server 10.228.147.34 --port 26547
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument('--jumanpp_command', dest='jumanpp_command', type=str, action='store', default="/mnt/violet/share/tool/juman++v2/bin/jumanpp")
    parser.add_argument("--server", default=None, type=str, required=True,
                        help="server IP address.")
    parser.add_argument("--port", default=None, type=int, required=True,
                        help="server port.")
    args = parser.parse_args()    
    main(args)
