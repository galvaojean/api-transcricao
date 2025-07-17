from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai, os, tempfile

app = Flask(__name__)
CORS(app)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Rota que serve a interface
@app.route("/")
def home():
    return render_template("index.html")

# Rota de transcrição
@app.route("/transcrever", methods=["POST"])
def transcrever():
    if 'audio' not in request.files:
        return jsonify({"erro": "campo 'audio' não enviado"}), 400

    f = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        f.save(tmp.name)
        path = tmp.name

    try:
        with open(path, "rb") as af:
            resp = openai.Audio.transcribe("whisper-1", af)
        txt = resp.get("text", "")
    except Exception as e:
        return jsonify({"erro": str(e)}), 500
    finally:
        os.remove(path)

    return jsonify({"transcricao": txt})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
