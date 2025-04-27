from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import time
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ThingSpeak configuration
THINGSPEAK_WRITE_API_KEY = "0Y4ASEUMTBHFQ8I5"
THINGSPEAK_URL = "https://api.thingspeak.com/update"

class SensorData(BaseModel):
    green: float
    red: float

# Video streaming functions
def generate_frames():
    video_path = "videos/sample_video.mp4"
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = video.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 0.03
    
    while True:
        success, frame = video.read()
        
        if not success:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(delay)

# Video streaming endpoint
@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(generate_frames(),
                           media_type='multipart/x-mixed-replace; boundary=frame')

# ThingSpeak endpoint
@app.post("/send-data/")
async def send_data_to_thingspeak(data: SensorData):
    try:
        # Tạo payload để gửi đến ThingSpeak
        payload = {
            'api_key': THINGSPEAK_WRITE_API_KEY,
            'field1':int( data.green),  # field1: green
            'field2': int(data.red)     # field2: red
        }
        
        # Gửi request đến ThingSpeak
        response = requests.get(THINGSPEAK_URL, params=payload)
        
        if response.status_code == 200 and response.text != '0':
            return {
                "status": "success",
                "message": "Dữ liệu đã được gửi thành công",
                "entry_id": response.text
            }
        else:
            raise HTTPException(status_code=500, detail="Không thể gửi dữ liệu đến ThingSpeak")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Server is running with video streaming and ThingSpeak API"}