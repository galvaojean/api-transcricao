import os
import time
import tempfile
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import openai
import assemblyai as aai

# inicialização
app = Flask(__name__)
CORS(app)

# carrega chaves do ambiente
openai.api_key = os.getenv("OPENAI_API_KEY")
AAI = aai.Client()  # o SDK lê ASSEMBLYAI_API_KEY automaticamente

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcrever", methods=["POST"])
def transcrever():
    # 1) valida
    if "audio" not in request.files:
        return jsonify({"erro": "campo 'audio' não enviado"}), 400

    audio_file = request.files["audio"]

    # 2) salva em arquivo temporário
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio_path = tmp.name
    tmp.close()
    audio_file.save(audio_path)

    try:
        # 3) upload + diarização
        up = AAI.upload(audio_path)
        tx = AAI.transcript.create(audio_url=up["upload_url"], speaker_labels=True)
        tx_id = tx["id"]

        # 4) espera ficar pronto
        while True:
            st = AAI.transcript.get(tx_id)
            if st["status"] in ("completed", "error"):
                break
            time.sleep(2)

        if st["status"] == "error":
            return jsonify({"erro": st.get("error", "erro desconhecido")}), 500

        full_text  = st["text"]
        utterances = st.get("utterances", [])

        # 5) resumo + insights
        summary, insights = gerar_resumo_insights(full_text)

    finally:
        os.remove(audio_path)

    return jsonify({
        "transcricao": full_text,
        "diarizacao": utterances,
        "resumo": summary,
        "insights": insights
    })


def gerar_resumo_insights(texto: str):
    prompt = f"""
Você é um assistente. Receba esta transcrição:

{texto}

1) Resuma em até 5 frases.
2) Liste 5 bullets de próximos passos ou insights.

Responda _apenas_ com JSON:
{{"resumo":"...", "insights":["...","...",...]}}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    content = resp.choices[0].message.content.strip()
    data = json.loads(content)
    return data["resumo"], data["insights"]


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
