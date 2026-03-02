import os
import torch
from groq import Groq
from diffusers import StableDiffusionPipeline
from PIL import Image

# =========================
# LOAD API KEY FROM ENV
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY as environment variable.")

client = Groq(api_key=GROQ_API_KEY)

# =========================
# GET IMAGE PATH
# =========================
image_path = input("Enter path of your drawing image: ")
image = Image.open(image_path)
image.show()

# =========================
# IDENTIFY DRAWING USING GROQ
# =========================
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You identify hand-drawn sketches."},
        {"role": "user", "content": "Identify the object in this drawing in one short sentence."}
    ],
    temperature=0.3
)

drawing_description = response.choices[0].message.content
print("AI thinks this is:", drawing_description)

# =========================
# LOAD STABLE DIFFUSION
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "runwayml/stable-diffusion-v1-5"

if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id
    ).to(device)

# =========================
# GENERATE ANIMATION
# =========================
final_prompt = f"cartoon style animated image of {drawing_description}, colorful, pixar style"

result_image = pipe(final_prompt).images[0]
result_image.save("animated_output.png")

print("Animated image saved as animated_output.png")
