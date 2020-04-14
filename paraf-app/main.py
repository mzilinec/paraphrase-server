#!/usr/bin/env python

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from websocket import create_connection

import os
import re
import subprocess
import time
import json

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
import serving_utils
from tensor2tensor.utils import hparam
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir
import tensorflow.compat.v1 as tf

DATA_DIR = os.environ.get("DATA_DIR", "/opt/models/serving")
LANGUAGES = os.environ.get("LANGUAGES", "de en es hu lt ru").split()

TF_SERVING_ADDRESS = os.environ.get("TF_SERVING_HOST", "127.0.0.1:9000")

app = Flask("paraphrase-server")
CORS(app)

# Getting languages from uedin model:
# less 37-to-37/en-centric/exp01/vocab.multi.yml | grep '<2' | sed -r -E "s/<2([a-z][a-z])>.*/\1/" | sed -r -E 's/.*/"\0",/' | tr '\n' ' '

##### TENSORFLOW CODE START #####

def make_request_fn():
    request_fn = serving_utils.make_grpc_request_fn(
        servable_name="transformer",
        server=TF_SERVING_ADDRESS,
        timeout_secs=10)
    return request_fn


def start_tf():
    import problems
    global problem
    global request_fn
    problem = registry.problem("translate_many_to_many")
    hparams = hparam.HParams(data_dir=DATA_DIR)  # ./vocab.txt
    problem.get_hparams(hparams)
    request_fn = make_request_fn()
    print("TF initialized OK")


def _tf_predict(inputs):
    outputs = serving_utils.predict(inputs, problem, request_fn)
    out = []
    for prediction in outputs:
        foo, scores = prediction
        out.append(foo)
    return out

##### TENSORFLOW CODE END   #####

def do_translation(inputs):
    return _tf_predict(inputs)

def preprocess(text):
    return text.replace("\n", " ").replace("\0", "").strip()

@app.route("/translate", methods=['POST', 'GET'])
def on_request():
    if request.method == 'POST':
        data = request.get_json(force=True)
        source_text = preprocess(data.get("text"))
        source_lang = data.get("lang")
    else:
        source_text = request.args.get('text')
        source_lang = request.args.get('lang')
    if not source_text or not source_lang:
        return "Please provide the following parameters: text, lang", 400
    return translate(source_text, source_lang)

def _token_for_language(lang):
    if lang not in LANGUAGES:
        raise Exception()
    return 2 + LANGUAGES.index(lang)

def translate(source_text, source_lang):
    # translate text to all languages
    inputs = []
    for language in LANGUAGES:
        try:
            lang_token = _token_for_language(language)
        except:
            return 400, jsonify({"status": "400", "message": "Invalid language: %s" % source_lang})
        inputs.append((source_text, lang_token))
    translated = do_translation(inputs)

    # translate text back to source language
    inputs = []
    for lang, sent in zip(LANGUAGES, translated):
        try:
            lang_token = _token_for_language(source_lang)
        except:
            return 400, jsonify({"status": "400", "message": "Invalid language: %s" % source_lang})
        inputs.append((sent, lang_token))
    translated = do_translation(inputs)

    return jsonify(dict(zip(LANGUAGES, translated)))

start_tf()
