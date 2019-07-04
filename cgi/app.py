# -*- coding: utf-8 -*-
import xmlrpc.client as xmlrpc_client

from bottle import Bottle, HTTPResponse, request
import zenhan

from ip_conf import host, port


app = Bottle()


@app.get('/')
def json():
    body = {"status": "OK", "message": "こんにちは世界"}
    r = HTTPResponse(status=200, body=body)
    r.set_header("Content-Type", "application/json")
    return r


@app.post('/post')
def sentiment_analysis():
    sentence = request.json['utterance']
    sentence = zenhan.h2z(sentence)
    sentiment_analysis_client = xmlrpc_client.ServerProxy(f'http://{host}:{port}')
    prediction = sentiment_analysis_client.get_prediction(sentence)  # 1(Pos) or -1(Neg)

    if prediction == 1:
        result = 'Positive'
    elif prediction == -1:
        result = 'Negative'
    else:
        result = 'None'

    body = {"status": "OK", "result": result}
    r = HTTPResponse(status=200, body=body)
    r.set_header("Content-Type", "application/json")
    return r


# @app.route('/echo', method='post')
# def echo():
#     utterance = request.json['hello']
#     body = {"status": "OK", "message": utterance}
#     r = HTTPResponse(status=200, body=body)
#     r.set_header("Content-Type", "application/json")
#     return r
