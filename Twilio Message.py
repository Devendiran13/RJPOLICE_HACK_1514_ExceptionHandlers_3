#Immediate message to the district officer when the target class is detected.
from twilio.rest import Client

account_sid = 'your_account_sid'
auth_token = 'your_auth_token'

twilio_phone_number = 'your_twilio_phone_number'
destination_phone_number = 'destination_phone_number'

client = Client(account_sid, auth_token)
def send_twilio_message():
    message_body = "Target class detected!"
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone_number,
        to=destination_phone_number
    )
    print(f"Message sent with SID: {message.sid}")
def process_image(image):
    if detect_target_class(image):
        send_twilio_message()
