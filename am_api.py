import os.path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from AM.api import speech2pinyin
from flask import Flask, request

app = Flask(__name__)


@app.route("/audio", methods=['POST'])
def parseAudio():
    file = request.files.get('wav')
    filepath = os.path.join("tmp", file.filename)
    file.save(filepath)
    answer = speech2pinyin(filepath)
    os.remove(filepath)
    return {"pinyin": answer}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=False)
