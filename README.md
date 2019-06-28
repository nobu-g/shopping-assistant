# BERT-masked-lm-demo

A demo for masked lm (language model) using BERT.

## server
    $ cd server
	$ python server.py --bert_model /larch/shibata/bert/preprocess/181213_subword_wikipedia/pretraining_model_20e --server 10.228.147.34 --port 26547

- Specify `server` and `port`.

## cgi
	$ cp cgi/* /home/(username)/public_html/somewhere/
	
- Specify `host` and `port` in `masked_lm_conf.py`, which are the same as ones specified when you run `server.py`.
- Access `http://lotus.kuee.kyoto-u.ac.jp/~(username)/somewhere/index.cgi`.
    - Example: http://lotus.kuee.kyoto-u.ac.jp/~shibata/bert/masked_lm/index.cgi

