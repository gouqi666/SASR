from flask import Flask, request
from LM.api import pinyin2text
import torch
from pypinyin import pinyin

app = Flask(__name__)


@app.route("/text", methods=['POST'])
def convertPinyin():
    _pinyin = eval(request.form.get("pinyin"))
    return {"text": pinyin2text(_pinyin)}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=False)
