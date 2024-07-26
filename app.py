from flask import Flask, request, jsonify
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.google_gemini import GoogleGemini
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

llm = GoogleGemini(api_key=os.getenv('AIzaSyDGRc7hBixnbur-4uxYLVEepe8RgQsn8ac'))
pandas_ai = None

@app.route('/upload', methods=['POST'])
def upload_file():
    global pandas_ai
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if file and file.filename.endswith(('.csv', '.xlsx')):
            file_path = os.path.join("/path/to/save", file.filename)  
            file.save(file_path)
            df = pd.read_csv(file_path) if file.filename.endswith('.csv') else pd.read_excel(file_path)
            pandas_ai = SmartDataframe(df, config={"llm": llm, "verbose": True})
            return jsonify({"message": "File successfully uploaded and SDF created"}), 200
        else:
            return jsonify({"error": "Invalid file format"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/interact', methods=['POST'])
def interact():
    global pandas_ai
    if pandas_ai is None:
        return jsonify({"error": "No file uploaded yet"}), 400

    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "No message provided"}), 400

    try:
        result = pandas_ai.chat(message)
        return jsonify({"response": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

@app.route('/file-check', methods=['GET'])
def file_check():
    global pandas_ai
    return jsonify({"file_present": pandas_ai is not None}), 200

if __name__ == '__main__':
    app.run(debug=True)