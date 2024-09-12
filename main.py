from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from fastapi.responses import FileResponse
import io
import json
import os
import re
import traceback
import shutil
import numpy as np
import face_recognition
from deepface import DeepFace
import cv2
from pydub import AudioSegment
import os
import shutil
import math
from deepgram import DeepgramClient, PrerecordedOptions
import os
import markdown
import pdfkit
import os
from groq import Groq

app = FastAPI()
client = Groq(
    api_key='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'
)

# The API key we created in step 3
DEEPGRAM_API_KEY = '82578abdf9477547c395bd610e8109a9700789bc'

def reset_save_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def audio_spilter(audio):
    directory = 'audio_directory'
    reset_save_directory(directory)
    song = AudioSegment.from_mp3(audio)
    ten_minutes = 10 * 60 * 1000
    audio_length = len(song) / 60000
    print(f'Audio length: {math.floor(audio_length)} minutes')      
    count = 0
    file_paths = []
    for x in range(0, len(song), ten_minutes):
        count += 1
        audio_segment = song[x:x+ten_minutes]
        file_path = os.path.join(os.path.abspath(directory), f"audio_split_{count}.mp3")
        audio_segment.export(file_path, format="mp3")
        file_paths.append(file_path)

    print('Number of splits:', count)
    return file_paths

def speech_to_text(audio_path):
    paths = audio_spilter(audio_path)
    transcript = None
    
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    for x in paths:
        print(x)
        with open(f'{x}', mode='rb') as buffer_data:
            payload = { 'buffer': buffer_data }

            options = PrerecordedOptions(
                smart_format=True, model="nova-2", language="en-GB"
            )

            response = deepgram.listen.rest.v('1').transcribe_file(payload, options)
            if transcript:
                print(len(transcript.split(' ')))
                transcript = transcript + response.results.channels[0].alternatives[0].transcript
            else:
                transcript = response.results.channels[0].alternatives[0].transcript

    return transcript

def notes_gen(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                    "role": "system",
                    "content": "You are an expert academic writer tasked with producing highly detailed and structured notes based on a provided text. From the given text, extract the key concepts, topics, and themes, and then elaborate on each topic comprehensively. Structure the notes using clear headings, subheadings, and bullet points for clarity. Ensure that the notes are thorough, insightful, and enriched with additional relevant knowledge beyond the text, showcasing deep understanding. The final output should be written in markdown format and consist of 3000 words, with all explanations logically organized and expanded, going beyond the provided content. Focus on delivering advanced, informative, and well-articulated explanations that stand on their own.",

                "role": "user",
                "content": text,
            }
        ],
        model="mixtral-8x7b-32768",
    )

    return chat_completion.choices[0].message.content

def pdf_gen(audio_path, output_dir):
    text = speech_to_text(audio_path)
    notes = notes_gen(text)
    html_text = markdown.markdown(notes)
    pdf_file_path = os.path.join(output_dir, 'lstm_notes.pdf')
    pdfkit.from_string(html_text, pdf_file_path)
    return pdf_file_path


with open('/Users/naveenpoliasetty/Downloads/David2.0/david2.0/cnn_faces_data.json', 'r') as f:
    data = json.load(f)

embeddings = [x['Face_embedding'] for x in data]
rolls = [x['Roll'] for x in data]

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm1 * norm2) if norm1 and norm2 else 0.0

def cnn_model(path):
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image, model='hog')
    return face_recognition.face_encodings(image, face_locations)

def find_face(image_path, directory_path):
    pattern = r'\/([A-Za-z0-9]+)\.jpg'
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path)]
    res_li = []

    for x in file_paths:
        match = re.search(pattern, x)
        if match:
            extracted_string = match.group(1)
            res = DeepFace.verify(image_path, x, model_name="ArcFace", detector_backend='yolov8')

            if res['verified']:
                res_li.append({extracted_string: res['distance']})
    
    if not res_li:
        return {'roll': 'Face not found'}

    least_entry = min(res_li, key=lambda item: next(iter(item.values())), default={})
    least_key = next(iter(least_entry), 'Face not found')
    print('roll :',least_key)

    return {'roll': least_key}

def reset_save_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cosine = np.dot(v1, v2)/ (norm1*norm2)
    return cosine

def extract_frames_per_second(video_path, output_folder):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      print("Error opening video!")
      return
  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_count = 0
  extracted_frame_count = 0
  while True:
      ret, frame = cap.read()
      if not ret:
          print("Can't receive frame (stream end?). Exiting...")
          break
      if frame_count % int(fps) == 0:
          filename = f"{output_folder}/frame_{extracted_frame_count}.jpg"
          cv2.imwrite(filename, frame)
          extracted_frame_count += 1
      frame_count += 1
  cap.release()
  print(f"Extracted {extracted_frame_count} frames to {output_folder}")
  return output_folder

def extract_faces(input_dir, output_dir='crop_face'):
    image_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir)]
    reset_save_directory(output_dir)
    count = 0
    less_pixel_count  = 0
    for index1, image_path in enumerate(image_paths):
        face_objs = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend='yolov8' 
        )
        image = cv2.imread(image_path)

        for face_obj in face_objs:
            x = face_obj['facial_area']['x']
            y = face_obj['facial_area']['y']
            w = face_obj['facial_area']['w']
            h = face_obj['facial_area']['h']
            cropped_face = image[y:y+h, x:x+w]
            confidence = face_obj['confidence']
            if confidence >= 0.8:
                output_path = os.path.join(output_dir, f'cropped_face_{count}.jpg')
                count += 1
                cv2.imwrite(output_path, cropped_face)
                print(f"Confidence : {confidence} and Pixel quality : {w*h} Saved cropped face at: {output_path}")
            else:
                less_pixel_count += 1
    print(f'Ignored {less_pixel_count} faces due to less resolution')  
    frame_paths = [f'/Users/naveenpoliasetty/Downloads/David2.0/david2.0/crop_face/{x}' for x in os.listdir('/Users/naveenpoliasetty/Downloads/David2.0/david2.0/crop_face')]
    frame_embeddings = []
    for x in frame_paths:
        dfs = DeepFace.represent(
            img_path = x,
        )
        frame_embeddings.append(dfs[0]['embedding'])
    with open('/Users/naveenpoliasetty/Downloads/David2.0/david2.0/face_embeddings.json','r')as f:
        face_embeddings = json.load(f)
    presenties = []
    for x in frame_embeddings:
        val = []
        keys = []
        for key, value in face_embeddings.items():
            simi = float(cosine_similarity(x, value))
            val.append(simi)
            keys.append(key)
        index = np.argmax(val)
        if val[index] >= 0.4:
            presenties.append(keys[index])
    return {'Presenties':presenties}

SAVE_DIRECTORY = "uploaded_images"

def reset_save_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

reset_save_directory(SAVE_DIRECTORY)

@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    try:
        if not image.filename.lower().endswith(".jpg"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPG images are accepted.")
        
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data))
        
        file_path = os.path.join(SAVE_DIRECTORY, image.filename)
        image_pil.save(file_path, format='JPEG')

        directory_path = '/Users/naveenpoliasetty/Downloads/David2.0/david2.0/faces'
        result = find_face(file_path, directory_path)

        return result

    except Exception as e:
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        raise HTTPException(status_code=400, detail=error_details)
    
SAVE_DIRECTORY_VIDEO = 'attendance'
if not os.path.exists(SAVE_DIRECTORY_VIDEO):
    os.makedirs(SAVE_DIRECTORY_VIDEO)  # Ensure the directory exists


@app.post("/upload-video/")
async def upload_video(video: UploadFile = File(...)):
    try:
        file_path = os.path.join(SAVE_DIRECTORY_VIDEO, video.filename)

        with open(file_path, 'wb') as f:  # Open in binary mode
            f.write(await video.read())  # Write the binary content to the file
        SAVE_DIRECTORY_FRAMES = 'api_frames'
        reset_save_directory(SAVE_DIRECTORY_FRAMES)
        extract_frames_per_second(file_path, SAVE_DIRECTORY_FRAMES)
        res = extract_faces(SAVE_DIRECTORY_FRAMES)
        return res

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"There was an error processing the video: {str(e)}")
    
SAVE_DIRECTORY_AUDIO = 'speech'
if not os.path.exists(SAVE_DIRECTORY_AUDIO):
    os.makedirs(SAVE_DIRECTORY_AUDIO)

@app.post("/upload-audio/")
async def upload_audio(audio: UploadFile = File(...)):
    try :
        file_path = os.path.join(SAVE_DIRECTORY_AUDIO, audio)
        with open(file_path, 'wb')as f:
            f.write(await audio.read())
        SAVE_DIRECTORY_AUDIO = 'audio_lecture'
        reset_save_directory(SAVE_DIRECTORY_AUDIO)
        
        SAVE_DIRECTORY_PDF = 'notes'
        reset_save_directory(SAVE_DIRECTORY_PDF)
        pdf_path = pdf_gen(file_path, SAVE_DIRECTORY_PDF)
        return FileResponse(pdf_path, media_type='application/pdf', filename='lstm_notes.pdf')
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'There was an error processsing the audio:{str(e)}')


    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
