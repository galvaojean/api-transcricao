import os
import time
import tempfile
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
import assemblyai as aai

# Inicializa Flask e CORS
app = Flask(__name__, template_folder="templates")
CORS(app)

# Chaves de API (definidas como variáveis de ambiente no Render)
openai.api_key = os.getenv("OPENAI_API_KEY")
AAI = aai.Client()  # o SDK já lê ASSEMBLYAI_API_KEY do env

@app.route("/")
def home():
    # Serve o front-end
    return render_template("index.html")

@app.route("/transcrever", methods=["POST"])
def transcrever():
    # 1) Verifica se o áudio veio
    if "audio" not in request.files:
        return jsonify({"erro": "campo 'audio' não enviado"}), 400

    # 2) Salva em arquivo temporário
    audio_file = request.files["audio"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio_path = tmp.name
    tmp.close()
    audio_file.save(audio_path)

    try:
        # 3) Upload + diarização na AssemblyAI
        upload_resp = AAI.upload(audio_path)
        tx       = AAI.transcript.create(
            audio_url       = upload_resp["upload_url"],
            speaker_labels  = True
        )
        transcript_id = tx["id"]

        # 4) Polling até completar
        while True:
            status = AAI.transcript.get(transcript_id)
            if status["status"] in ("completed", "error"):
                break
            time.sleep(2)

        if status["status"] == "error":
            return jsonify({"erro": status.get("error", "erro desconhecido")}), 500

        full_text  = status["text"]
        utterances = status.get("utterances", [])

        # 5) Gera resumo + insights com gpt-3.5-turbo
        prompt = f"""
Você é um assistente que recebe a transcrição completa de uma reunião:

{full_text}

1) Forneça um resumo executivo em até 5 frases.
2) Liste 5 insights ou próximos passos em formato de bullets.

Responda apenas em JSON, no formato:
{{ "resumo": "...", "insights": ["...", "...", ...] }}
"""
        ai_resp = openai.ChatCompletion.create(
            model       = "gpt-3.5-turbo",
            messages    = [{"role":"user","content":prompt}],
            temperature = 0.3
        )
        content = ai_resp.choices[0].message.content.strip()
        data    = json.loads(content)
        resumo  = data.get("resumo", "")
        insights= data.get("insights", [])

    finally:
        # 6) Limpa o temporário
        try: os.remove(audio_path)
        except: pass

    # 7) Retorna tudo
    return jsonify({
        "transcricao": full_text,
        "diarizacao": utterances,
        "resumo":      resumo,
        "insights":    insights
    })

if __name__ == "__main__":
    # Porta obrigatória no Render é obtida via $PORT
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
