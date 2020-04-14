# Paraphrasing server

This repository contains a server for generating distinct paraphrases using round-trip multilingual neural machine translation.

## Installation
* Obtain a trained multilingual NMT model from https://github.com/mzilinec/many-to-many-nmt
* Install tensorflow serving (CPU/GPU) from https://github.com/tensorflow/serving \
On Ubuntu, CPU version can also be installed by
```sudo apt install tensorflow_serving```
* Install Python requirements
```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running from the command line
* Start tensorflow serving
```tensorflow_model_server --port=9000 --model_name=transformer --model_base_path=$PWD/export```
* Start REST paraphrasing server
```
cd paraf-app
export DATA_DIR="path to vocab file from your trained model"
export LANGUAGES="space delimited languages from vocab, in original order"
export FLASK_APP="main.py"
flask run
```

## Running as a service
* Set the model directory and CUDA path in systemd/tfserving.service
* Set the environment variables and port in systemd/uwsgi-paraf.service
* Copy the systemd files
```
cp systemd/tfserving.service /etc/systemd/system/tfserving.service && \
cp systemd/uwsgi-paraf.service /etc/systemd/system/uwsgi-paraf.service && \
cp systemd/uwsgi-paraf.socket /etc/systemd/system/uwsgi-paraf.socket
```
* Enable the services
```
systemctl daemon-reload && \
systemctl enable tfserving.service && \
systemctl enable uwsgi-paraf.service && \
systemctl enable uwsgi-paraf.socket
```

## Usage
The server supports GET and POST with two parameters, `text` and `lang`. \
The request
```bash
curl -H 'Content-Type: application/json' \
     -XPOST -d \
     '{"text": "Hi, how are you doing today?", "lang": "en"}' \
     localhost:5000/translate
```
will produce paraphrases from round-trip translation for each supported language:
```json
{
    "de":"Hello, how's it today?",
    "en":"Hi, how are you doing today?",
    "es":"Hello, how are you today?",
    "hu":"Hey, how are you today?",
    "lt":"Hello, how are you today?",
    "ru":"Hey, how are you doing today?"
}
```

## Disclaimer
This server was created at the [Institute of Formal and Applied Linguistics, Charles University](http://ufal.mff.cuni.cz/).
