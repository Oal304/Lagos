import re
from flask import current_app
from flask_mail import Message

def send_contact_email(name, email, message, mail):
    if not all([name, email, message]):
        return {'error': 'All fields are required.'}

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return {'error': 'Invalid email address.'}

    try:
        # Email to support team
        support_msg = Message(
            subject='Contact Form Submission',
            sender=current_app.config['MAIL_DEFAULT_SENDER'],  # Get sender email from app config
            recipients=['c.ola@alustudent.com']  # Support team email address
        )
        support_msg.body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        mail.send(support_msg)
        
        # Email to customer
        customer_msg = Message(
            subject='We have received your message',
            sender=current_app.config['MAIL_DEFAULT_SENDER'],  # The same sender email
            recipients=[email]  # Customer's email address
        )
        customer_msg.body = f"Dear {name},\n\nWe have received your message: '{message}'. We will contact you within two business days.\n\nBest regards,\nLagosFlow Team"
        mail.send(customer_msg)

        return {'success': 'Message sent successfully.'}
    except Exception as e:
        return {'error': str(e)}
