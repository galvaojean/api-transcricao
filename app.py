from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/transcrever', methods=['POST'])
def transcrever():
    audio = request.files['audio']
    audio.save("audio.webm")

    audio_file = open("audio.webm", "rb")
    transcription = openai.Audio.transcribe("whisper-1", audio_file)

    texto = transcription['text']

    os.remove("audio.webm")

    return jsonify({"transcricao": texto})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
