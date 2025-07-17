import os
import time
import tempfile
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import openai
import assemblyai as aai

# Inicialização
app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
AAI = aai.Client(os.getenv("ASSEMBLYAI_API_KEY"))

@app.route("/")
def home():
    # Serve a interface estática em templates/index.html
    return render_template("index.html")

@app.route("/transcrever", methods=["POST"])
def transcrever():
    # 1) Recebe o áudio enviado pelo cliente
    if "audio" not in request.files:
        return jsonify({"erro": "campo 'audio' não enviado"}), 400

    audio_file = request.files["audio"]
    # salva em arquivo temporário
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio_path = tmp.name
    audio_file.save(audio_path)
    tmp.close()

    try:
        # 2) Envia para AssemblyAI com diarização de speakers
        upload_resp = AAI.upload(audio_path)
        tx = AAI.transcript.create(
            audio_url=upload_resp["upload_url"],
            speaker_labels=True
        )
        transcript_id = tx["id"]

        # 3) Polling até a transcrição ficar pronta
        while True:
            status = AAI.transcript.get(transcript_id)
            if status["status"] in ("completed", "error"):
                break
            time.sleep(2)

        if status["status"] == "error":
            return jsonify({"erro": status.get("error", "erro desconhecido")}), 500

        full_text = status["text"]
        utterances = status.get("utterances", [])

        # 4) Gera resumo e insights via OpenAI
        summary, insights = gerar_resumo_insights(full_text)

    finally:
        # 5) Cleanup do arquivo temporário
        try:
            os.remove(audio_path)
        except OSError:
            pass

    # 6) Retorna o JSON com tudo
    return jsonify({
        "transcricao": full_text,
        "diarizacao": utterances,
        "resumo": summary,
        "insights": insights
    })

def gerar_resumo_insights(texto_completo: str):
    prompt = f"""
Você é um assistente que recebe a transcrição completa de uma reunião:

{texto_completo}

1) Forneça um resumo executivo em até 5 frases.
2) Liste 5 insights ou próximos passos em formato de bullets.

Responda apenas com um JSON no formato:
{{ "resumo": "...", "insights": ["...", "...", ...] }}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    content = resp.choices[0].message.content.strip()
    data = json.loads(content)
    return data["resumo"], data["insights"]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
