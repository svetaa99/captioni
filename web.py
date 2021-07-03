from flask import Flask, jsonify
from testing_caption_generator import generate_caption

app = Flask(__name__)


@app.route("/generate-caption/<img_url>", methods=['GET'])
def hello_world(img_url):
    desc = generate_caption(img_url)
    response = jsonify({'description': desc})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
