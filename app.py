import streamlit as st
import re
import io
import random
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
import torch

# Load pre-trained models from Hugging Face with caching
@st.cache_resource
def load_models():
    # Image captioning model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    # Text generation model
    story_generator = pipeline("text-generation", model="distilgpt2")
    return processor, caption_model, story_generator

# Extract descriptive caption from the uploaded image
def get_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=30)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption.lower().strip()

# Core logic to build a randomized story based on image and user settings
def generate_story(caption, generator, mood, child_name):
    name = child_name if child_name else "a little friend"
    
    # Story templates categorized by mood
    mood_data = {
        "Funny 🤡": {
            "openings": [
                "In a silly place where socks like to dance,", "In a town where it rains chocolate cookies,",
                "In a world where all the trees are made of green jelly,", "High up in a cloud that tastes like lemon cake,",
                "In a funny garden where flowers tell jokes,"
            ],
            "atmospheres": [
                "Everyone was giggling and doing silly moon-jumps.", "The air smelled like sweet bubblegum and laughter.",
                "Tiny bubbles of joy were floating everywhere.", "Even the sun was wearing big, funny sunglasses.",
                "The grass was tickling everyone's feet!"
            ],
            "closings": [
                "What a funny day! Everyone went to sleep laughing.", "Goodnight! Don't let the tickle-monsters get you!",
                "It was the silliest day ever recorded in history.", "Everyone dreamed of dancing pineapples that night."
            ]
        },
        "Adventurous 🚀": {
            "openings": [
                "On a big, brave journey to the moon,", "Deep inside a secret jungle full of mystery,",
                "Across the sparkling blue sea on a magic boat,", "At the top of a mountain that touches the stars,",
                "In a hidden cave filled with glowing crystals,"
            ],
            "atmospheres": [
                "It was time for a grand discovery!", "The air was full of exciting magic and power.",
                "Every step felt like a new and wonderful surprise.", "The wind was cheering for the brave explorers.",
                "A bright map appeared in the sky to show the way."
            ],
            "closings": [
                "The brave explorers were ready for another mission tomorrow.", "What a grand adventure! The stars cheered for them.",
                "They returned home feeling like the bravest heroes ever.", "A new map was already waiting for their next trip."
            ]
        },
        "Cozy 🧸": {
            "openings": [
                "In a warm house with a soft, crackling fire,", "Under a big, fluffy blanket of stars,",
                "In a quiet garden where the moon is a night-light,", "Inside a tiny cottage made of sweet dreams,",
                "In a peaceful meadow where the wind sings lullabies,"
            ],
            "atmospheres": [
                "Everything felt as warm as a big, soft hug.", "The air was quiet and smelled like fresh milk.",
                "The world was soft and gentle, like a sleeping kitten.", "Tiny fireflies were dancing to keep everyone company.",
                "A golden light made everything look peaceful and kind."
            ],
            "closings": [
                "Everyone felt safe and warm, dreaming of happy things.", "Goodnight, little friend. Sleep tight in your cozy bed.",
                "The moon watched over them with a loving smile.", "It was a day full of peace that ended with a soft kiss."
            ]
        }
    }
    
    data = mood_data[mood]
    
    # Random character actions
    adventures = [
        f"{name} was looking for a secret treasure of joy.",
        f"{name} was making new friends in this magical place.",
        f"{name} was learning how to fly with invisible wings.",
        f"{name} was sharing a very special secret today.",
        f"{name} was helping everyone find their way home."
    ]
    
    # Contextual introductions for the image content
    caption_intros = [
        f"Suddenly, {name} saw {caption} right in front of them!",
        f"The most amazing thing happened when {caption} appeared.",
        f"Everyone stopped to look because {caption} was so beautiful.",
        f"It was a lucky day to find {caption} in the magic world.",
        f"Look! {caption} is joining the fun today!"
    ]

    # Use AI to generate a creative bridge sentence
    prompt = f"In this {mood} story, {caption} was special because"
    ai_result = generator(prompt, max_new_tokens=20, do_sample=True, temperature=0.8)[0]['generated_text']
    ai_part = ai_result.replace(prompt, "").strip().split('.')[0].split(',')[0]
    
    # Fallback for inappropriate or short AI content
    if len(ai_part) < 5 or any(w in ai_part.lower() for w in ["work", "born", "die", "parent"]):
        ai_part = "it was filled with the magic of friendship and love"

    # Randomly select a story structure for variety
    structure_type = random.choice([1, 2])
    
    if structure_type == 1:
        segments = [
            random.choice(data["openings"]),
            random.choice(adventures),
            random.choice(caption_intros),
            f"It felt wonderful because {ai_part}.",
            random.choice(data["atmospheres"]),
            random.choice(data["closings"])
        ]
    else:
        segments = [
            f"One day, {random.choice(adventures)}",
            random.choice(data["openings"]),
            f"Everything changed when they saw {caption}.",
            f"It was like {ai_part}, and everyone started to cheer!",
            random.choice(data["atmospheres"]),
            random.choice(data["closings"])
        ]
    
    full_story = " ".join(segments)
    words = full_story.split()
    
    # Ensure story length stays within 50-100 words
    if len(words) > 100:
        full_story = " ".join(words[:95]) + "."
    elif len(words) < 55:
        full_story += f" {name} knew that this was the start of something truly magical."
        
    return full_story

# Convert the generated story text to audio
def text_to_speech(text):
    tts = gTTS(text=text, lang="en", slow=False)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Magic Storyteller", page_icon="📖", layout="centered")
st.title("✨ Magic Storyteller for Kids ✨")

# Sidebar for personalization settings
with st.sidebar:
    st.header("🌟 Story Settings")
    child_name = st.text_input("What is your name?", placeholder="Enter name here...")
    mood = st.selectbox("How should the story feel?", ["Cozy 🧸", "Funny 🤡", "Adventurous 🚀"])
    st.info("Tip: Every story is unique! Try different moods.")

st.markdown(f"Hello **{child_name if child_name else 'Friend'}**! Upload an image to start your **{mood}** story.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    # Main execution button
    if st.button("🎭 Tell me my story", type="primary"):
        with st.spinner("Creating your magical story..."):
            # Process image and generate story
            processor, caption_model, story_generator = load_models()
            caption = get_caption(image, processor, caption_model)
            
            st.subheader("🖼️ I see:")
            st.info(caption)
            
            story = generate_story(caption, story_generator, mood, child_name)
            
            st.subheader(f"📖 A {mood} Story for {child_name if child_name else 'you'}")
            st.success(story)
            
            # Generate and play audio
            audio = text_to_speech(story)
            st.subheader("🔊 Listen to the story")
            st.audio(audio, format="audio/mp3")
            st.balloons()
