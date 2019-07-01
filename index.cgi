#!/mnt/berry/home/ueda/public_html/shopping-assistant/.venv/bin/python
# -*- coding: utf-8 -*-

import sys
import io
import os
import cgi
import zenhan
import xmlrpc.client as xmlrpc_client
from masked_lm_conf import host, port, DEFAULT_SENTENCE


def print_form(sentence):
    print('<form>')
    print('<input name="sentence" size="120" value="{}" />'.format((sentence if sentence is not None else "")))
    print('<input type="submit" value="解析">')
    print('</form>')


def main():
    print("Content-Type: text/html")
    print()

    f = cgi.FieldStorage()
    sentence = f.getfirst('sentence', DEFAULT_SENTENCE)
        
    if sentence is not None:
        sentence = zenhan.h2z(sentence)
        
    print_form(sentence)

    # masked_lm_client = xmlrpc_client.ServerProxy('http://{}:{}'.format(host, port))
    # # prediction = masked_lm_client.get_predictions(sentence)
    # pid = os.getpid()
    # filename = "json/prediction_{}.json".format(pid)
    # with open(filename, mode='w', encoding='utf-8') as outfile:
    #     outfile.write(prediction)


if __name__ == "__main__":
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
