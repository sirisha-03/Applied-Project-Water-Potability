from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf # type: ignore
import pandas as pd
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

app = Flask(__name__)

ann_model = tf.keras.models.load_model("my_ann_model_v2.keras")
print("✅ Keras model loaded successfully")
with open("standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    print("✅ Scaler loaded successfully")
    print("Scaler mean:", scaler.mean_)
    print("Scaler scale:", scaler.scale_)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    try:
        data = request.json
        features = [
            data.get("ph"),
            data.get("hardness"),
            data.get("solids"),
            data.get("chloramines"),
            data.get("sulfate"),
            data.get("conductivity"),
            data.get("organic_carbon"),
            data.get("trihalomethanes"),
            data.get("turbidity")
        ]
        try:
            features_array = np.array([float(f) for f in features], dtype=np.float32).reshape(1, -1)
        except ValueError:
            return jsonify({"error": "All inputs must be numeric"}), 400

        features_scaled = scaler.transform(features_array)
        prob = ann_model.predict(features_scaled)[0][0]
        
        print("Received features:", features)
        print("Scaled features:", features_scaled)
        print("Predicted probability:", prob)

        
        threshold = 0.4
        prediction = 1 if prob > threshold else 0
        potability = "Potable" if prediction == 1 else "Not Potable"
        return jsonify({"potability": potability, "confidence": f"{prob * 100:.2f}%"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message")
        print("[INFO] Received message:", user_input)

        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick",
            messages=[
                {"role": "system", "content": (
                    "You are an expert on water potability, purification methods, and water quality parameters. "
                    "Answer clearly and accurately. If asked anything not related to water, politely decline."
                )},
                {"role": "user", "content": user_input}
            ],
            max_tokens=200,
            temperature=0.7
        )

        final_response = response.choices[0].message.content
        print("[INFO] AI Response:", final_response)

        return jsonify({"response": final_response})
    
    except Exception as e:
        print("[ERROR] Chat generation failed:", str(e))
        return jsonify({"error": f"Chat generation failed: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
