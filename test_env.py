from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("✅ HF_TOKEN loaded successfully:")
    print("🔐", hf_token[:10] + "...")
else:
    print("❌ HF_TOKEN not found! Check your .env file.")

