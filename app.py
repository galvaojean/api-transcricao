import os
import time
import tempfile
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
import assemblyai as aai

# Inicialização do Flask e CORS
app = Flask(__name__)
CORS(app)

# 1) Configura as chaves de API
openai.api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/transcrever", methods=["POST"])
def transcrever():
    # 2) Valida presença do áudio
    if "audio" not in request.files:
        return jsonify({"erro": "campo 'audio' não enviado"}), 400

    # 3) Salva áudio em arquivo temporário
    audio_file = request.files["audio"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    audio_path = tmp.name
    audio_file.save(audio_path)
    tmp.close()

    try:
        # 4) Upload + diarização na AssemblyAI
        transcriber = aai.Transcriber()
        with open(audio_path, "rb") as f:
            upload_url = transcriber.upload_file(f.read())

        config = aai.TranscriptionConfig(speaker_labels=True)
        transcript = transcriber.transcribe(upload_url, config=config)

        full_text = transcript.text
        utterances = [
            {"speaker": u.speaker, "start": u.start, "end": u.end, "text": u.text}
            for u in (transcript.utterances or [])
        ]

        # 5) Gera resumo e insights via OpenAI
        summary, insights = gerar_resumo_insights(full_text)

    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        # 6) Limpa arquivo temporário
        try: os.remove(audio_path)
        except OSError: pass

    # 7) Retorna tudo em JSON
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

Responda apenas em JSON, no formato:
{{"resumo":"...","insights":["...","...",...]}}
"""
    # Nova interface para openai>=1.0.0
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    content = resp.choices[0].message.content.strip()
    data = json.loads(content)
    return data.get("resumo",""), data.get("insights",[])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
