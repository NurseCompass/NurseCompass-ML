import os
from flask import Flask, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import config

app = Flask(__name__)

# Initialize Flask-Limiter to protect heavy AI routes from spam/DoS
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Removed try-except trap so the server reveals any missing dependencies
from routes.chatbot_routes import chatbot_bp
from routes.prediction_routes import prediction_bp
from routes.assessment_routes import assessment_bp

app.register_blueprint(chatbot_bp)
app.register_blueprint(prediction_bp)
app.register_blueprint(assessment_bp)
print("✅ All ML routes registered successfully!")

@app.route("/")
def home():
    return jsonify({"message": "NurseCompass ML Engine is Live!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)