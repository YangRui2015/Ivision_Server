import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import io
import torch

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from test_demo import get_score, rank

app = Flask(__name__)


@app.route("/score", methods=['POST'])
def score_route():
    openid = request.headers["user"]
    path = request.headers["path"]
    image = Image.open(path).convert("RGB")
    score = get_score(image)
    present_best_score, your_rank = rank(openid, score)
    string = str(score) + " " + str(present_best_score) + " " + str(your_rank)
    return string



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8081, debug=True)
