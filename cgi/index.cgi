#!/orange/ubrew/data/bin/python3
# -*- coding: utf-8 -*-
# import cgitb
#
#
# cgitb.enable(format='text')

activator = "/mnt/berry_f/home/ueda/work/shopping-assistant/.venv/bin/activate_this.py"
with open(activator) as f:
    exec(f.read(), {"__file__": activator})

from wsgiref.handlers import CGIHandler
from app import app

CGIHandler().run(app)
