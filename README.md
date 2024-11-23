# MeSure Backend only

## spin up this server by doing this steps:

1. clone the repo with PAT.
2. install git if not installed
3. install python nad python-env
4. make a python venv
5. pip install -r requirements.txt
6. sudo apt-get update
7. sudo apt-get install -y libgl1
8. run gunicorn -w 2 -k gevent --timeout 120 -b 0.0.0.0:8080 cgt_mesure_api_dev:app