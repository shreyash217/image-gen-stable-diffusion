from dotenv import load_dotenv
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import platform

# Load token from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Debug: Print token (just first 10 chars)
print("🔐 Token:", hf_token[:10] if hf_token else "Token not found!")

# Check token
if not hf_token:
    raise ValueError("❌ HF_TOKEN not found in .env file!")

# Prompt input
prompt = input("Enter your image prompt: ")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Load the model pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    use_auth_token=hf_token,
    torch_dtype=torch.float32
).to(device)

# Generate image
image = pipe(prompt).images[0]

# Save image
output_path = "generated_image.png"
image.save(output_path)
print(f"✅ Image saved as {output_path}")

# Open image (OS-specific)
if platform.system() == "Windows":
    os.system(f"start {output_path}")
elif platform.system() == "Darwin":  # macOS
    os.system(f"open {output_path}")
else:
    print("📂 Open 'generated_image.png' from the file explorer.")
