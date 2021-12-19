import os.path

from flask import Flask, request
import requests

app = Flask(__name__)


@app.route("/", methods=['POST'])
def speech2text():
    audio = request.files.get("audio")
    file_path = os.path.join("cache", audio.filename)
    audio.save(file_path)
    pinyin = requests.post("http://localhost:6000/audio", files={"wav": open(file_path, 'rb')}).json()[
        'pinyin']
    text = requests.post("http://localhost:6001/text", data={"pinyin": str(pinyin)}).json()['text']
    return {"data": text}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9108, debug=True)
