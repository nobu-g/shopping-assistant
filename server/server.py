import sys
import argparse
import xmlrpc.server as xmlrpc_server

from sentiment_analysis import SentimentAnalysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert-model', default=None, type=str, required=True,
                        help='path to BERT model directory')
    parser.add_argument('--fine-tuned-model', default=None, type=str, required=True,
                        help='path to fine-tuned PosNeg classifier model file')
    parser.add_argument('--jumanpp-command', type=str, action='store',
                        default="/mnt/violet/share/tool/juman++v2/bin/jumanpp")
    parser.add_argument("--server", default=None, type=str, required=True,
                        help="server IP address.")
    parser.add_argument("--port", default=None, type=int, required=True,
                        help="server port.")
    args = parser.parse_args()
    server = xmlrpc_server.SimpleXMLRPCServer((args.server, args.port), allow_none=True)

    sa_model = SentimentAnalysis(args.bert_model, args.fine_tuned_model, args.jumanpp_command)
    server.register_function(sa_model.get_prediction, 'get_prediction')
    print("loading done.", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
