import os
import time
import tempfile
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import openai
import assemblyai as aai

# ─── 1) Inicialização ───────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
CORS(app)

# as chaves devem estar configuradas em Environment Variables do Render:
openai.api_key = os.getenv("OPENAI_API_KEY")
AAI = aai.Client()  # o SDK já lê ASSEMBLYAI_API_KEY da variável de ambiente

# ─── 2) Rota da interface ──────────────────────────────────────────────────────

@app.route("/")
def home():
    # templates/index.html
    return render_template("index.html")

# ─── 3) Rota de transcrição ────────────────────────────────────────────────────

@app.route("/transcrever", methods=["POST"])
def transcrever():
    # 3.1) Valida se veio áudio
    if "audio" not in request.files:
        return jsonify({"erro": "campo 'audio' não enviado"}), 400

    audio_file = request.files["audio"]

    # 3.2) Salva num arquivo temporário
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio_path = tmp.name
    tmp.close()
    audio_file.save(audio_path)

    try:
        # 3.3) Upload + diarização
        up = AAI.upload(audio_path)
        tx = AAI.transcript.create(
            audio_url      = up["upload_url"],
            speaker_labels = True
        )
        tx_id = tx["id"]

        # 3.4) Polling até ficar pronto
        while True:
            status = AAI.transcript.get(tx_id)
            if status["status"] in ("completed", "error"):
                break
            time.sleep(2)

        if status["status"] == "error":
            return jsonify({"erro": status.get("error", "desconhecido")}), 500

        full_text  = status["text"]
        utterances = status.get("utterances", [])

        # 3.5) Gera resumo + insights
        resumo, insights = gerar_resumo_insights(full_text)

    finally:
        # 3.6) Remove o arquivo temporário
        try:
            os.remove(audio_path)
        except OSError:
            pass

    # 3.7) Retorna JSON completo
    return jsonify({
        "transcricao": full_text,
        "diarizacao":  utterances,
        "resumo":       resumo,
        "insights":    insights
    })


# ─── 4) Função de resumo via OpenAI ────────────────────────────────────────────

def gerar_resumo_insights(texto: str):
    prompt = f"""
Você é um assistente que recebe a transcrição completa de uma reunião:
{texto}

1) Resuma em até 5 frases.
2) Liste 5 próximos passos ou insights em formato de bullets.

Responda apenas com um JSON no formato:
{{"resumo":"...","insights":["...","...", ...]}}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    content = resp.choices[0].message.content.strip()
    data = json.loads(content)
    return data["resumo"], data["insights"]


# ─── 5) Entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
