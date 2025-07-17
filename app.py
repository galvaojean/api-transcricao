from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
import tempfile

app = Flask(__name__)
CORS(app)

# pega a chave da OpenAI da variável de ambiente configurada no Render
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route("/", methods=["GET"])
def home():
    return "API de Transcrição online. Envie áudio via POST /transcrever (campo 'audio')."

# Aceita POST na raiz e em /transcrever (pra facilitar)
@app.route("/", methods=["POST"])
@app.route("/transcrever", methods=["POST"])
def transcrever():
    if 'audio' not in request.files:
        return jsonify({"erro": "campo 'audio' não encontrado no upload"}), 400

    audio_storage = request.files['audio']

    # Salva o arquivo temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        temp_path = tmp.name
        audio_storage.save(temp_path)

    try:
        with open(temp_path, "rb") as af:
            resp = openai.Audio.transcribe("whisper-1", af)
        texto = resp.get("text", "")
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return jsonify({"transcricao": texto})
    

if __name__ == "__main__":
    # Render ignora isso quando usa gunicorn, mas deixa aqui pra rodar local
    app.run(host="0.0.0.0", port=10000)
