import os
from dotenv import load_dotenv
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ HF_TOKEN not found in .env file!")

# Setup model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
    use_auth_token=hf_token,
    torch_dtype=torch.float32
).to(device)

# Generator function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    return "generated_image.png"

# Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt", placeholder="A valley at sunset, Ghibli style..."),
    outputs=gr.File(label="Download Image"),
    title="🖼️ Stable Diffusion Generator",
    description="Enter a text prompt to generate an image. Click 'Download Image' to save it."
)

# Launch app
interface.launch()
