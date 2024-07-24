import os
import cv2
import time
import pygame
import base64
import requests
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from datetime import datetime
import mss

# Load environment variables
load_dotenv() 

# Initialize OpenAI and ElevenLabs clients
openAI_client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def text_to_speech_stream(text: str):
    # Perform the text-to-speech conversion and stream audio
    response = client.text_to_speech.convert(
        voice_id="a8p00hpqmTpR1cLnk76X",  # voice_id is found on elevenlabs.com
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    audio_stream = BytesIO()

    # Create the generated_audio directory if it doesn't exist
    output_dir = "generated_audio"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a unique filename based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_dir, f"audio_{timestamp}.mp3")

    # Open a file to save the audio data
    with open(output_file_path, "wb") as output_file:
        # Collect all chunks from the generator
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)
                output_file.write(chunk)  # Save the chunk to the file

    # Reset stream position to the beginning
    audio_stream.seek(0)
    
    # Load and play the audio
    pygame.mixer.music.load(audio_stream, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    print(f"Audio saved to {output_file_path}")

def ImageToText(image_path="frame.jpg", type="picture"):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}"
    }

    prompt = {
        "picture" : """You are Kylie J. (the influencer) and you help me with taking the perfect Instagram photo. With some short sentences, 
                          say what you like and don't like and give tips what to change in the photo to go viral. Say something about what you see in the photo. 
                          Say the following in your own words at the ends: 'Ok let's try another one, please press space to snap a photo'. 
                          I only want maximum 3 sentences, and write what Kylie would say in the voice of a funny Aussie man. Be funny, and no emojis in output text as it is fed to elevenlabs voice models. Dont hallucinate.""",
        "screenshot": """You are Kylie J. (the influencer) and you help me with taking the perfect Instagram photo of a google meets meeting. With some short sentences, 
                          say what you like and don't like and give tips what to change in the photo to go viral. Say something about what you see in the photo, but keep the focus on the people, and feel free to call out people by their first names. 
                          Say the following in your own words at the ends: 'Ok let's try another one, please press s to take another screenshot'. 
                          I only want maximum 3 sentences, and write what Kylie would say in the voice of a funny Aussie man. Be funny, and no emojis in output text as it is fed to elevenlabs voice models. Dont hallucinate"""
              }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[type]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 128
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    data = response.json()
    content = data['choices'][0]['message']['content']
    return content

def get_frame(vid):
    ret, frame = vid.read()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    return frame

def display_image(image_path):
    image = pygame.image.load(image_path)
    image = pygame.transform.scale(image, (1920, 1080))
    screen.blit(image, (0, 0))
    pygame.display.update()


# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Set up the drawing window
screen = pygame.display.set_mode([1920, 1080])
pygame.display.set_caption('Camera Feed')

# Initialize video capture
vid = cv2.VideoCapture(0)

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                print("-> Played iPhone shutter sound")
                ret, frame = vid.read()
                resized_frame = cv2.resize(frame, (1280, 720), fx=0.35, fy=0.35) 
                if ret:
                    pygame.mixer.music.load("iPhone_shutter.mp3")
                    pygame.mixer.music.play()
                    cv2.imwrite('frame.jpg', resized_frame)
                    display_image('frame.jpg')

                    # Generate advice based on the captured image
                    advice = ImageToText(image_path="frame.jpg", type="picture")
                    print("-> Generated advice")
                    print(advice)

                    print("-> playing cloned voice")
                    text_to_speech_stream(advice)
                    
            elif event.key == pygame.K_s:
                # Take a screenshot of the main screen
                pygame.mixer.music.load("iPhone_shutter.mp3")
                pygame.mixer.music.play()
                with mss.mss() as sct:
                    screenshot = sct.shot(output="screenshot.png")
                    print("Screenshot saved as screenshot.png")
                display_image('screenshot.png')
                advice = ImageToText(image_path="screenshot.png", type="screenshot")
                print("-> Generated advice")
                print(advice)

                print("-> playing cloned voice")
                text_to_speech_stream(advice)

            if event.key == pygame.K_q:
                print("-> Exiting")
                running = False

    frame_surface = get_frame(vid)
    if frame_surface:
        screen.blit(frame_surface, (0, 0))
    pygame.display.update()

vid.release()
cv2.destroyAllWindows()
pygame.quit()
print("Exited")