#!/mnt/berry_f/home/shibata/anaconda3/envs/python3-web/bin/python
# -*- coding: utf-8 -*-

import sys
# import cgi
import json
import zenhan
import xmlrpc.client as xmlrpc_client
from ip_conf import host, port


def main():
    print('Content-Type: text/plain')
    print()

    # print("<TITLE>CGI script output</TITLE>")
    # print("<H1>This is my first CGI script</H1>")
    print('Hello, world!')

    raw_json = sys.stdin.read()
    input_data = json.loads(raw_json)
    # raw_json['key1'] = 'hoge'
    # print()
    # print(json.dumps(raw_json))
    # print()

    input_data = zenhan.h2z(input_data)

    sentiment_analysis_client = xmlrpc_client.ServerProxy(f'http://{host}:{port}')
    prediction = sentiment_analysis_client.get_prediction(input_data)  # 1(Pos) or -1(Neg)

    if prediction == 1:
        print('Positive')
    elif prediction == -1:
        print('Negative')
    else:
        print('None')

    # pid = os.getpid()
    # filename = "json/prediction_{}.json".format(pid)
    # with open(filename, mode='w', encoding='utf-8') as outfile:
    #     outfile.write(prediction)


if __name__ == "__main__":
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
