import os
from flask import Flask, jsonify
import config

app = Flask(__name__)

# Register Blueprints safely
try:
    from routes.chatbot_routes import chatbot_bp
    from routes.prediction_routes import prediction_bp
    from routes.assessment_routes import assessment_bp

    app.register_blueprint(chatbot_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(assessment_bp)
    print("✅ All ML routes registered successfully!")
except Exception as e:
    print(f"❌ Error registering blueprints: {e}")

@app.route("/")
def home():
    return jsonify({"message": "NurseCompass ML Engine is Live!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)