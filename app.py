from bottle import Bottle, run, response, template
from pymongo import MongoClient
from datetime import datetime
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time

# Connect to MongoDB
client = MongoClient('mongodb+srv://leduckhuong2002:7321nho132@cluster0.m2szcpr.mongodb.net/dev')
db = client['dev'] 
collection = db['plates']

# Initialize Bottle app
app = Bottle()

# Load YOLO model
model = YOLO('best.pt')

# Initialize PaddleOCR with English model
ocr = PaddleOCR(use_angle_cls=True, lang='latin')

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera. Vui lòng kiểm tra kết nối và quyền truy cập camera.")
    exit()

# Variables to track the last seen plate and time
last_seen_time = None
plate_text_last_seen = None
first_detect_time = None

def gen_frames():
    global last_seen_time, plate_text_last_seen, first_detect_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        plate_detected = False

        for result in results:
            for box in result.boxes.data.cpu().numpy():
                x_min, y_min, x_max, y_max = map(int, box[:4])
                license_plate_region = frame[y_min:y_max, x_min:x_max]

                # Use PaddleOCR to recognize text
                ocr_result = ocr.ocr(license_plate_region, cls=True)

                if ocr_result is not None:
                    try:
                        plate_text = ''.join([word[1][0] for line in ocr_result for word in line])
                        plate_text = ''.join(filter(lambda char: char.isalnum() or char == ' ', plate_text)).replace(" ", "")
                        
                        plate_detected = True

                        if plate_text == plate_text_last_seen:
                            current_time = time.time()
                            if first_detect_time is None:
                                first_detect_time = current_time

                            if current_time - first_detect_time >= 10:
                                now = datetime.now()
                                readable_time = now.strftime('%Y-%m-%d %H:%M:%S')
                                
                                # Save plate and time to MongoDB
                                if last_seen_time is None:
                                    data = {
                                        'plate_text': plate_text,
                                        'timestamp_in': readable_time,
                                        'timestamp_out': None,
                                        'paid': False,
                                        'slot':1
                                    }
                                    collection.insert_one(data)
                                    last_seen_time = readable_time

                        else:
                            plate_text_last_seen = plate_text
                            first_detect_time = None

                        # Draw bounding box and plate text on the frame
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Lỗi xảy ra khi nhận dạng ký tự: {e}")

        if not plate_detected and last_seen_time is not None:
            now = datetime.now()
            readable_time_out = now.strftime('%Y-%m-%d %H:%M:%S')
            
            # Update exit time in MongoDB
            collection.update_one(
                {'plate_text': plate_text_last_seen, 'timestamp_out': None},
                {'$set': {'timestamp_out': readable_time_out}}
            )
            
            # Reset variables
            plate_text_last_seen = None
            first_detect_time = None
            last_seen_time = None

        # Convert frame to MJPEG and return
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f4f4f4;
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .content {
            display: flex;
            align-items: flex-start;
            margin-top: 20px;
        }
        .video-container {
            margin-right: 30px;
        }
        .price-container {
            font-size: 24px;
            color: #d9534f;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }
        .qr-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .qr-container img {
            width: 150px;
            height: 150px;
            border: 2px solid #333;
            padding: 5px;
            border-radius: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <h1>License Plate Recognition from Camera</h1>
    <div class="content">
        <div class="video-container">
            <img src="/video_feed" width="640" height="480" alt="Video Feed">
        </div>
        <div class="qr-container">
            <div class="price-container">
                QR Code for payment
            </div>
           <a href="http://127.0.0.1:3000/revenue/payment"> <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=http://127.0.0.1:8080/payment_info" alt="QR Code"><a/>
        </div>
    </div>
 
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    return gen_frames()




if __name__ == '__main__':
    run(app, host='0.0.0.0', port=8000, debug=True)
