#!/mnt/berry_f/home/shibata/anaconda3/envs/python3-web/bin/python
# -*- coding: utf-8 -*-

import cgi
# import zenhan
# import xmlrpc.client as xmlrpc_client
# from masked_lm_conf import host, port, DEFAULT_SENTENCE


def main():
    print("Content-Type: text/html")
    print()

    print("<TITLE>CGI script output</TITLE>")
    print("<H1>This is my first CGI script</H1>")
    print("Hello, world!")

    # f = cgi.FieldStorage()
    # sentence = f.getfirst('sentence', DEFAULT_SENTENCE)
        
    # if sentence is not None:
    #     sentence = zenhan.h2z(sentence)
        
    # masked_lm_client = xmlrpc_client.ServerProxy('http://{}:{}'.format(host, port))
    # # prediction = masked_lm_client.get_predictions(sentence)
    # pid = os.getpid()
    # filename = "json/prediction_{}.json".format(pid)
    # with open(filename, mode='w', encoding='utf-8') as outfile:
    #     outfile.write(prediction)


if __name__ == "__main__":
    # sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
