import firebase_admin
from firebase_admin import credentials, auth
import os

# Initialize Firebase Admin
cred = credentials.Certificate("firebase-admin-key.json")
firebase_admin.initialize_app(cred)

def verify_token(token: str):
    """Verify Firebase ID token"""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        return None