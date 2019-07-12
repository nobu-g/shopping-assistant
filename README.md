# shopping-assistant

CGI server for Dialogflow application "shopping-assistant"

## server
    $ cd server
    $ python server.py --bert-model /mnt/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE --fine-tuned-model /home/ueda/work/shopping-assistant/server/model.pth --server 10.228.147.34 --port 26547

- Specify `server` and `port`.

## cgi
	$ cp cgi/* /home/(username)/public_html/somewhere/

- Specify `host` and `port` in `ip_conf.py`, which are the same as ones specified when you run `server.py`.
- Access `http://lotus.kuee.kyoto-u.ac.jp/~(username)/somewhere/index.cgi/post`.

