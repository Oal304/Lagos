# run.py

import sys
import os
from flask import Flask
from flask_cors import CORS
from flask_mail import Mail
from aws_lambda_powertools.event_handler import api_gateway
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

# Add the backend directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import routes
from backend.routes.predict import predict_bp
from backend.routes.newsletter import newsletter_bp
from backend.routes.contact import contact_blueprint

# Initialize the Flask app
app = Flask(__name__)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
db = SQLAlchemy(app)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "https://lagosflowbucket.s3-website-us-east-1.amazonaws.com"
        ],  # production domain here
        "methods": ["GET", "POST", "PUT", "DELETE"],  # Specify the methods allowed
        "headers": ["Content-Type", "Authorization"],  # Specify allowed headers
        "supports_credentials": True
    }
})

# Configure Flask-Mail
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv('MAIL_USERNAME'),
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD'),
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_DEFAULT_SENDER')
)


# Initialize Flask-Mail
mail = Mail(app)

# Register blueprints
app.register_blueprint(predict_bp, url_prefix='/api')
app.register_blueprint(newsletter_bp, url_prefix='/api')
app.register_blueprint(contact_blueprint, url_prefix='/api')


@app.route('/')
def home():
    return "Welcome to the LagosFlow Traffic Prediction API!"

# Set up the API Gateway handler
api = api_gateway.ApiGatewayResolver()

def lambda_handler(event, context):
    """
    AWS Lambda handler function to process events and contexts.
    """
    return api.resolve(event, context)

# Run the Flask app locally
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
