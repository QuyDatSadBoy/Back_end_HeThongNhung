from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import time
import numpy as np
import threading
import requests
import os
import json
from datetime import datetime
import logging
from ultralytics import YOLO
from onvif import ONVIFCamera

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("traffic_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrafficSystem")

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

# Cấu hình camera và model
CAMERA_IP = "192.168.43.81"
CAMERA_USERNAME = "admin"
CAMERA_PASSWORD = "L2A17879"
RTSP_PORT = 554
ONVIF_PORT = 80
CAMERA_CHANNEL = 1
CAMERA_STREAM = 1  # Sử dụng substream cho hiệu suất tốt hơn

# Đường dẫn RTSP
RTSP_URL = f"rtsp://{CAMERA_USERNAME}:{CAMERA_PASSWORD}@{CAMERA_IP}:{RTSP_PORT}/cam/realmonitor?channel={CAMERA_CHANNEL}&subtype={CAMERA_STREAM}"

# Cấu hình cho model
class Config:
    # Đường dẫn tới model YOLOv8n
    DEFAULT_MODEL = "yolov8n.pt"
    
    # Kích thước resize frame trước khi đưa vào model
    INFERENCE_SIZE = (320, 320)  # Giảm kích thước để tăng tốc độ
    
    # Tần suất thực hiện phát hiện đối tượng
    DETECTION_INTERVAL = 5  # Chỉ phát hiện mỗi 5 frames để giảm tải CPU
    
    # Ngưỡng tin cậy cho phát hiện đối tượng
    CONFIDENCE_THRESHOLD = 0.3
    
    # Thời gian ở mỗi hướng camera (giây)
    DIRECTION_SWITCH_INTERVAL = 15
    
    # Thời gian tối thiểu để đèn xanh
    MIN_GREEN_TIME = 10
    
    # Thời gian tối đa cho đèn xanh
    MAX_GREEN_TIME = 30
    
    # Thời gian cho đèn vàng
    YELLOW_TIME = 3

# Các model Pydantic
class SensorData(BaseModel):
    green: float
    red: float

class TrafficLightControl(BaseModel):
    horizontal: str  # "red", "yellow", "green"
    vertical: str  # "red", "yellow", "green"
    horizontal_time: int  # thời gian còn lại (giây)
    vertical_time: int  # thời gian còn lại (giây)

class CameraDirection(BaseModel):
    direction: str  # "horizontal" hoặc "vertical"

class TrafficStats(BaseModel):
    horizontal_count: int
    vertical_count: int
    horizontal_density: float
    vertical_density: float
    current_direction: str

# Biến toàn cục để lưu trạng thái
frame_global = None
processed_frame_global = None
current_direction = "horizontal"
is_direction_switching = False
traffic_light_state = {
    "horizontal": "red",
    "vertical": "green",
    "horizontal_time": 0,
    "vertical_time": 30
}
traffic_stats = {
    "horizontal_count": 0,
    "vertical_count": 0,
    "horizontal_density": 0.0,
    "vertical_density": 0.0,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
historical_data = {
    "horizontal": {
        "timestamps": [],
        "counts": [],
        "densities": []
    },
    "vertical": {
        "timestamps": [],
        "counts": [],
        "densities": []
    }
}

# Khóa để đồng bộ hóa truy cập vào biến toàn cục
frame_lock = threading.Lock()
stats_lock = threading.Lock()
traffic_light_lock = threading.Lock()

# Controller để điều khiển camera PTZ qua ONVIF
class ImouPTZController:
    def __init__(self, ip, port, username, password):
        """
        Khởi tạo controller để điều khiển camera IMOU PTZ
        
        Args:
            ip (str): Địa chỉ IP của camera
            port (int): Cổng ONVIF (thường là 80)
            username (str): Tên đăng nhập camera
            password (str): Mật khẩu camera
        """
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.camera = None
        self.ptz_service = None
        self.media_service = None
        self.profile_token = None
        self.connected = False
        
        # Định nghĩa 2 vị trí cố định
        self.positions = {
            "horizontal": {"pan": 0, "tilt": 0, "zoom": 1},  # Vị trí mặc định
            "vertical": {"pan": 90, "tilt": 0, "zoom": 1}    # Vị trí xoay 90 độ
        }
        
        # Cờ kiểm soát
        self.is_switching = False
        
        # Kết nối đến camera
        if self.connect_onvif():
            logger.info("Đã kết nối ONVIF thành công")
            # Thiết lập các preset
            self.setup_presets()
        else:
            logger.warning("Không thể kết nối ONVIF, sẽ sử dụng chế độ mô phỏng")
        
    def connect_onvif(self):
        """Kết nối camera qua ONVIF để điều khiển PTZ"""
        try:
            # Thử nhiều cách khác nhau để tìm WSDL
            wsdl_paths = [
                None,                       # Mặc định
                '/etc/onvif/wsdl/',         # Linux
                './wsdl/',                  # Thư mục hiện tại
                os.path.expanduser('~/.onvif/wsdl/')  # Thư mục home
            ]
            
            for wsdl_path in wsdl_paths:
                try:
                    if wsdl_path is None:
                        self.camera = ONVIFCamera(self.ip, self.port, self.username, self.password)
                    else:
                        self.camera = ONVIFCamera(self.ip, self.port, self.username, self.password, wsdl_dir=wsdl_path)
                    break
                except Exception as e:
                    logger.warning(f"Không thể kết nối với WSDL tại {wsdl_path}: {e}")
                    continue
            
            if self.camera is None:
                logger.error("Không thể tìm thấy WSDL phù hợp")
                return False
            
            # Lấy các service cần thiết
            self.media_service = self.camera.create_media_service()
            self.ptz_service = self.camera.create_ptz_service()
            
            # Lấy profile token - sử dụng profile đầu tiên
            profiles = self.media_service.GetProfiles()
            self.profile_token = profiles[0].token
            
            logger.info(f"Đã kết nối ONVIF thành công với camera {self.ip}")
            logger.info(f"Profile token: {self.profile_token}")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Lỗi kết nối ONVIF: {e}")
            self.connected = False
            return False

    def setup_presets(self):
        """Thiết lập các preset cho camera"""
        if not self.connected:
            return
            
        try:
            # Thiết lập preset cho vị trí horizontal (mặc định)
            logger.info("Di chuyển camera về vị trí mặc định (horizontal)")
            self.move_absolute(
                self.positions["horizontal"]["pan"],
                self.positions["horizontal"]["tilt"],
                self.positions["horizontal"]["zoom"]
            )
            time.sleep(3)  # Đợi camera di chuyển xong
            
            # Lưu preset
            self.set_preset(1, "horizontal")
            logger.info("Đã lưu preset 1 (horizontal)")
            
            # Thiết lập preset cho vị trí vertical (xoay 90 độ)
            logger.info("Di chuyển camera đến vị trí vertical (xoay 90 độ)")
            self.move_absolute(
                self.positions["vertical"]["pan"],
                self.positions["vertical"]["tilt"],
                self.positions["vertical"]["zoom"]
            )
            time.sleep(3)  # Đợi camera di chuyển xong
            
            # Lưu preset
            self.set_preset(2, "vertical")
            logger.info("Đã lưu preset 2 (vertical)")
            
            # Quay về vị trí mặc định
            self.goto_preset(1)
            
            return True
        except Exception as e:
            logger.error(f"Lỗi thiết lập preset: {e}")
            return False

    def move_absolute(self, pan, tilt, zoom):
        """Di chuyển camera đến vị trí tuyệt đối"""
        if not self.connected:
            return False
            
        try:
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = self.profile_token
            
            request.Position = {
                'PanTilt': {
                    'x': pan,
                    'y': tilt
                },
                'Zoom': {
                    'x': zoom
                }
            }
            
            self.ptz_service.AbsoluteMove(request)
            logger.info(f"Đã di chuyển camera đến vị trí Pan={pan}, Tilt={tilt}, Zoom={zoom}")
            return True
        except Exception as e:
            logger.error(f"Lỗi di chuyển camera: {e}")
            return False

    def set_preset(self, preset_id, preset_name):
        """Lưu vị trí hiện tại thành preset"""
        if not self.connected:
            return False
            
        try:
            request = self.ptz_service.create_type('SetPreset')
            request.ProfileToken = self.profile_token
            request.PresetName = preset_name
            request.PresetToken = str(preset_id)
            
            response = self.ptz_service.SetPreset(request)
            logger.info(f"Đã lưu preset {preset_id} ({preset_name})")
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu preset: {e}")
            return False

    def goto_preset(self, preset_id):
        """Di chuyển camera đến preset đã lưu"""
        if not self.connected:
            return False
            
        try:
            request = self.ptz_service.create_type('GotoPreset')
            request.ProfileToken = self.profile_token
            request.PresetToken = str(preset_id)
            request.Speed = {'PanTilt': {'x': 1.0, 'y': 1.0}, 'Zoom': {'x': 1.0}}
            
            self.ptz_service.GotoPreset(request)
            logger.info(f"Đã di chuyển đến preset {preset_id}")
            
            # Cập nhật hướng hiện tại
            global current_direction
            if preset_id == 1:
                current_direction = "horizontal"
            elif preset_id == 2:
                current_direction = "vertical"
                
            return True
        except Exception as e:
            logger.error(f"Lỗi di chuyển đến preset: {e}")
            return False

    def move_to_direction(self, direction):
        """Di chuyển camera đến hướng cụ thể"""
        global current_direction, is_direction_switching
        
        if direction not in ["horizontal", "vertical"]:
            logger.error(f"Hướng không hợp lệ: {direction}")
            return False
            
        # Nếu đã ở hướng cần chuyển, không cần làm gì
        if direction == current_direction and not is_direction_switching:
            logger.info(f"Camera đã ở hướng {direction}")
            return True
            
        # Đánh dấu đang chuyển hướng
        is_direction_switching = True
        logger.info(f"Đang chuyển camera từ {current_direction} sang {direction}")
        
        success = False
        if not self.connected:
            # Chế độ mô phỏng
            logger.info(f"Mô phỏng: Chuyển hướng sang {direction}")
            time.sleep(2)  # Giả lập thời gian di chuyển
            success = True
        else:
            # Di chuyển đến preset tương ứng
            preset_id = 1 if direction == "horizontal" else 2
            success = self.goto_preset(preset_id)
            time.sleep(2)  # Đợi camera di chuyển hoàn tất
        
        # Cập nhật hướng hiện tại
        if success:
            current_direction = direction
            
        # Đánh dấu đã hoàn thành chuyển hướng
        is_direction_switching = False
        return success

    def get_current_direction(self):
        """Trả về hướng hiện tại của camera"""
        global current_direction
        return current_direction

# Class phát hiện và đếm phương tiện
class VehicleDetector:
    def __init__(self, model_path=None):
        # Sử dụng mô hình YOLOv8n có sẵn nếu không cung cấp
        if model_path is None or not os.path.exists(model_path):
            self.model = YOLO(Config.DEFAULT_MODEL)
            logger.info(f"Đã tải model mặc định {Config.DEFAULT_MODEL}")
        else:
            self.model = YOLO(model_path)
            logger.info(f"Đã tải model từ {model_path}")
            
        # Các loại phương tiện cần phát hiện (theo danh sách COCO)
        self.vehicle_classes = {
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck',
            1: 'bicycle'
        }
        
        # Đếm số frame để giảm tần suất phát hiện
        self.frame_count = 0
        
        # Kết quả phát hiện gần nhất cho mỗi hướng
        self.last_detection_results = {
            "horizontal": None,
            "vertical": None
        }
        
        logger.info("Khởi tạo bộ phát hiện phương tiện thành công")
    
    def detect_vehicles(self, frame, direction):
        """
        Phát hiện phương tiện trong frame
        
        Args:
            frame: Khung hình từ camera
            direction (str): Hướng hiện tại của camera
            
        Returns:
            tuple: (frame với bounding boxes, số lượng phương tiện)
        """
        # Tăng đếm frame
        self.frame_count += 1
        
        # Nếu frame rỗng, trả về frame gốc và số lượng phương tiện là 0
        if frame is None:
            return frame, 0
        
        # Chỉ phát hiện đối tượng theo tần suất định trước để tiết kiệm CPU
        if self.frame_count % Config.DETECTION_INTERVAL == 0:
            # Resize frame để tăng tốc độ xử lý
            resized_frame = cv2.resize(frame, Config.INFERENCE_SIZE)
            
            # Thực hiện phát hiện đối tượng
            try:
                results = self.model(resized_frame, conf=Config.CONFIDENCE_THRESHOLD)
                
                # Lưu kết quả phát hiện cho hướng hiện tại
                self.last_detection_results[direction] = results[0]
            except Exception as e:
                logger.error(f"Lỗi khi phát hiện đối tượng: {e}")
                return frame, 0
        
        # Lấy kết quả phát hiện cho hướng hiện tại
        detection_result = self.last_detection_results[direction]
        
        # Nếu chưa có kết quả phát hiện, trả về frame gốc
        if detection_result is None:
            return frame, 0
        
        # Tạo bản sao của frame để vẽ lên
        annotated_frame = frame.copy()
        
        # Đếm phương tiện
        vehicle_count = 0
        
        # Tỷ lệ kích thước giữa frame gốc và frame đã resize
        scale_x = frame.shape[1] / Config.INFERENCE_SIZE[0]
        scale_y = frame.shape[0] / Config.INFERENCE_SIZE[1]
        
        # Vẽ bounding boxes và đếm phương tiện
        for box in detection_result.boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            
            # Kiểm tra xem đối tượng có phải là phương tiện không
            if cls_id in self.vehicle_classes and conf >= Config.CONFIDENCE_THRESHOLD:
                vehicle_type = self.vehicle_classes[cls_id]
                
                # Lấy tọa độ bounding box và điều chỉnh theo tỷ lệ
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                
                # Đếm phương tiện
                vehicle_count += 1
                
                # Vẽ bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Hiển thị loại phương tiện và độ tin cậy
                label = f"{vehicle_type}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hiển thị thông tin trên frame
        self.add_info_to_frame(annotated_frame, direction, vehicle_count)
        
        # Cập nhật số lượng và mật độ xe vào biến toàn cục
        self.update_traffic_stats(direction, vehicle_count)
        
        return annotated_frame, vehicle_count
    
    def calculate_density(self, vehicle_count):
        """
        Tính toán mật độ giao thông dựa trên số lượng phương tiện
        
        Args:
            vehicle_count (int): Số lượng phương tiện
            
        Returns:
            float: Mật độ giao thông (0-1)
        """
        # Số phương tiện tối đa dự kiến
        max_expected_vehicles = 15
        
        # Chuẩn hóa mật độ về khoảng 0-1
        density = min(vehicle_count / max_expected_vehicles, 1.0)
        
        return density
    
    def add_info_to_frame(self, frame, direction, count):
        """
        Thêm thông tin vào frame
        
        Args:
            frame: Khung hình cần thêm thông tin
            direction (str): Hướng hiện tại
            count (int): Số lượng phương tiện
        """
        # Hiển thị hướng
        cv2.putText(
            frame, 
            f"Direction: {direction}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 0, 0), 
            2
        )
        
        # Hiển thị tổng số phương tiện
        cv2.putText(
            frame, 
            f"Vehicles: {count}", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0), 
            2
        )
        
        # Tính mật độ giao thông
        density = self.calculate_density(count)
        
        # Hiển thị mật độ giao thông
        cv2.putText(
            frame, 
            f"Density: {density:.2f}", 
            (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 0, 0), 
            2
        )
    
    def update_traffic_stats(self, direction, count):
        """
        Cập nhật thống kê giao thông
        
        Args:
            direction (str): Hướng hiện tại
            count (int): Số lượng phương tiện
        """
        global traffic_stats, historical_data
        
        with stats_lock:
            # Cập nhật số lượng phương tiện
            if direction == "horizontal":
                traffic_stats["horizontal_count"] = count
                # Tính mật độ giao thông
                traffic_stats["horizontal_density"] = self.calculate_density(count)
            else:
                traffic_stats["vertical_count"] = count
                # Tính mật độ giao thông
                traffic_stats["vertical_density"] = self.calculate_density(count)
            
            # Cập nhật timestamp
            traffic_stats["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Thêm dữ liệu vào lịch sử
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if direction == "horizontal":
                historical_data["horizontal"]["timestamps"].append(current_time)
                historical_data["horizontal"]["counts"].append(count)
                historical_data["horizontal"]["densities"].append(self.calculate_density(count))
                
                # Giữ tối đa 100 điểm dữ liệu
                if len(historical_data["horizontal"]["timestamps"]) > 100:
                    historical_data["horizontal"]["timestamps"].pop(0)
                    historical_data["horizontal"]["counts"].pop(0)
                    historical_data["horizontal"]["densities"].pop(0)
            else:
                historical_data["vertical"]["timestamps"].append(current_time)
                historical_data["vertical"]["counts"].append(count)
                historical_data["vertical"]["densities"].append(self.calculate_density(count))
                
                # Giữ tối đa 100 điểm dữ liệu
                if len(historical_data["vertical"]["timestamps"]) > 100:
                    historical_data["vertical"]["timestamps"].pop(0)
                    historical_data["vertical"]["counts"].pop(0)
                    historical_data["vertical"]["densities"].pop(0)

# Lớp điều khiển đèn giao thông
class TrafficLightController:
    def __init__(self):
        self.default_times = {
            "horizontal": {
                "green": Config.MIN_GREEN_TIME,
                "yellow": Config.YELLOW_TIME,
                "red": Config.MIN_GREEN_TIME + Config.YELLOW_TIME
            },
            "vertical": {
                "green": Config.MIN_GREEN_TIME,
                "yellow": Config.YELLOW_TIME,
                "red": Config.MIN_GREEN_TIME + Config.YELLOW_TIME
            }
        }
        
        # Trạng thái hiện tại của đèn
        self.current_state = {
            "horizontal": "red",
            "vertical": "green",
            "horizontal_time": self.default_times["horizontal"]["red"],
            "vertical_time": self.default_times["vertical"]["green"]
        }
        
        # Thời điểm chuyển đổi đèn tiếp theo
        self.next_switch_time = time.time() + self.default_times["vertical"]["green"]
        
        # Giai đoạn hiện tại của chu kỳ đèn
        # 1: V-Green, H-Red
        # 2: V-Yellow, H-Red
        # 3: V-Red, H-Green
        # 4: V-Red, H-Yellow
        self.current_phase = 1
        
        # Flag để biết liệu hệ thống có đang chạy ở chế độ thông minh
        self.smart_mode_enabled = True
        
        logger.info("Khởi tạo bộ điều khiển đèn giao thông thành công")
    
    def update_traffic_lights(self, horizontal_density, vertical_density):
        """
        Cập nhật trạng thái đèn giao thông dựa trên mật độ giao thông
        
        Args:
            horizontal_density (float): Mật độ giao thông hướng ngang (0-1)
            vertical_density (float): Mật độ giao thông hướng dọc (0-1)
        """
        global traffic_light_state
        
        with traffic_light_lock:
            current_time = time.time()
            
            # Kiểm tra xem đã đến lúc chuyển đèn chưa
            if current_time >= self.next_switch_time:
                if self.current_phase == 1:
                    # V-Green, H-Red -> V-Yellow, H-Red
                    self.current_state = {
                        "horizontal": "red",
                        "vertical": "yellow",
                        "horizontal_time": self.default_times["vertical"]["yellow"],
                        "vertical_time": self.default_times["vertical"]["yellow"]
                    }
                    self.current_phase = 2
                    self.next_switch_time = current_time + self.default_times["vertical"]["yellow"]
                    
                elif self.current_phase == 2:
                    # V-Yellow, H-Red -> V-Red, H-Green
                    # Tính toán thời gian đèn xanh cho hướng ngang
                    if self.smart_mode_enabled:
                        green_time = self.calculate_green_time(horizontal_density)
                    else:
                        green_time = self.default_times["horizontal"]["green"]
                    
                    self.current_state = {
                        "horizontal": "green",
                        "vertical": "red",
                        "horizontal_time": green_time,
                        "vertical_time": green_time + self.default_times["horizontal"]["yellow"]
                    }
                    self.current_phase = 3
                    self.next_switch_time = current_time + green_time
                    
                elif self.current_phase == 3:
                    # V-Red, H-Green -> V-Red, H-Yellow
                    self.current_state = {
                        "horizontal": "yellow",
                        "vertical": "red",
                        "horizontal_time": self.default_times["horizontal"]["yellow"],
                        "vertical_time": self.default_times["horizontal"]["yellow"]
                    }
                    self.current_phase = 4
                    self.next_switch_time = current_time + self.default_times["horizontal"]["yellow"]
                    
                elif self.current_phase == 4:
                    # V-Red, H-Yellow -> V-Green, H-Red
                    # Tính toán thời gian đèn xanh cho hướng dọc
                    if self.smart_mode_enabled:
                        green_time = self.calculate_green_time(vertical_density)
                    else:
                        green_time = self.default_times["vertical"]["green"]
                    
                    self.current_state = {
                        "horizontal": "red",
                        "vertical": "green",
                        "horizontal_time": green_time + self.default_times["vertical"]["yellow"],
                        "vertical_time": green_time
                    }
                    self.current_phase = 1
                    self.next_switch_time = current_time + green_time
                
                # Cập nhật biến toàn cục
                traffic_light_state = self.current_state.copy()
                
                # Gửi dữ liệu đến ThingSpeak
                self.send_to_thingspeak()
                
                logger.info(f"Đèn giao thông đã chuyển: H={self.current_state['horizontal']}, V={self.current_state['vertical']}")
            else:
                # Cập nhật thời gian còn lại
                remaining_time = max(0, int(self.next_switch_time - current_time))
                
                if self.current_phase == 1:
                    self.current_state["vertical_time"] = remaining_time
                    self.current_state["horizontal_time"] = remaining_time + self.default_times["vertical"]["yellow"]
                elif self.current_phase == 2:
                    self.current_state["vertical_time"] = remaining_time
                    self.current_state["horizontal_time"] = remaining_time
                elif self.current_phase == 3:
                    self.current_state["horizontal_time"] = remaining_time
                    self.current_state["vertical_time"] = remaining_time + self.default_times["horizontal"]["yellow"]
                elif self.current_phase == 4:
                    self.current_state["horizontal_time"] = remaining_time
                    self.current_state["vertical_time"] = remaining_time
                
                # Cập nhật biến toàn cục
                traffic_light_state = self.current_state.copy()
    
    def calculate_green_time(self, density):
        """
        Tính toán thời gian đèn xanh dựa trên mật độ giao thông
        
        Args:
            density (float): Mật độ giao thông (0-1)
            
        Returns:
            float: Thời gian đèn xanh (giây)
        """
        # Thời gian tối thiểu và tối đa cho đèn xanh
        min_green_time = Config.MIN_GREEN_TIME
        max_green_time = Config.MAX_GREEN_TIME
        
        # Tính toán thời gian đèn xanh dựa trên mật độ
        green_time = min_green_time + (max_green_time - min_green_time) * density
        
        logger.info(f"Đã tính toán thời gian đèn xanh dựa trên mật độ {density:.2f}: {green_time:.1f} giây")
        return green_time
    
    def set_smart_mode(self, enabled):
        """Bật/tắt chế độ điều khiển thông minh"""
        self.smart_mode_enabled = enabled
        logger.info(f"Chế độ thông minh: {'Bật' if enabled else 'Tắt'}")
    
    def send_to_thingspeak(self):
        """Gửi dữ liệu trạng thái đèn giao thông đến ThingSpeak"""
        try:
            # Tạo payload
            payload = {
                'api_key': THINGSPEAK_WRITE_API_KEY,
                'field1': 1 if self.current_state["horizontal"] == "green" else 0,
                'field2': 1 if self.current_state["horizontal"] == "red" else 0
            }
            
            # Gửi request đến ThingSpeak
            response = requests.get(THINGSPEAK_URL, params=payload)
            
            if response.status_code == 200 and response.text != '0':
                logger.info(f"Đã gửi dữ liệu đến ThingSpeak: {payload}")
            else:
                logger.error(f"Không thể gửi dữ liệu đến ThingSpeak: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Lỗi khi gửi dữ liệu đến ThingSpeak: {e}")

# Khởi tạo các đối tượng
try:
    ptz_controller = ImouPTZController(CAMERA_IP, ONVIF_PORT, CAMERA_USERNAME, CAMERA_PASSWORD)
except Exception as e:
    logger.error(f"Lỗi khởi tạo PTZ controller: {e}")
    ptz_controller = None

try:
    vehicle_detector = VehicleDetector()
except Exception as e:
    logger.error(f"Lỗi khởi tạo Vehicle Detector: {e}")
    vehicle_detector = None

try:
    traffic_light_controller = TrafficLightController()
except Exception as e:
    logger.error(f"Lỗi khởi tạo Traffic Light Controller: {e}")
    traffic_light_controller = None

# Hàm để xử lý video stream
def process_video_stream():
    global frame_global, processed_frame_global, current_direction, is_direction_switching
    
    # Cấu hình OpenCV để tránh lỗi HEVC và giảm độ trễ
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024000|max_delay;0"
    
    # Kết nối đến camera qua RTSP
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer nhỏ để giảm độ trễ
    
    if not cap.isOpened():
        logger.error(f"Không thể kết nối đến camera tại {RTSP_URL}")
        return
    
    logger.info(f"Đã kết nối thành công đến camera tại {RTSP_URL}")
    
    # Thời gian chuyển đổi hướng camera
    last_direction_switch_time = time.time()
    
    try:
        while True:
            # Đọc frame từ camera
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Không thể đọc frame từ camera. Đang thử kết nối lại...")
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                continue
            
            # Lưu frame gốc vào biến toàn cục
            with frame_lock:
                frame_global = frame.copy()
            
            # Kiểm tra xem có cần chuyển đổi hướng camera không nếu không đang trong quá trình chuyển
            current_time = time.time()
            if current_time - last_direction_switch_time >= Config.DIRECTION_SWITCH_INTERVAL and not is_direction_switching:
                new_direction = "vertical" if current_direction == "horizontal" else "horizontal"
                
                # Tạo thread mới để xoay camera (không chặn luồng chính)
                def switch_direction():
                    if ptz_controller:
                        ptz_controller.move_to_direction(new_direction)
                    last_direction_switch_time = time.time()
                
                threading.Thread(target=switch_direction).start()
                last_direction_switch_time = current_time
            
            # Phát hiện và đếm phương tiện
            if vehicle_detector:
                processed_frame, vehicle_count = vehicle_detector.detect_vehicles(frame, current_direction)
                
                # Lưu frame đã xử lý vào biến toàn cục
                with frame_lock:
                    processed_frame_global = processed_frame.copy()
            
            # Cập nhật trạng thái đèn giao thông
            if traffic_light_controller:
                with stats_lock:
                    horizontal_density = traffic_stats["horizontal_density"]
                    vertical_density = traffic_stats["vertical_density"]
                
                traffic_light_controller.update_traffic_lights(horizontal_density, vertical_density)
            
            # Đợi một chút để giảm tải CPU
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Lỗi trong quá trình xử lý video: {e}")
    finally:
        cap.release()
        logger.info("Đã đóng kết nối camera")

# Bắt đầu luồng xử lý video
video_processing_thread = threading.Thread(target=process_video_stream, daemon=True)
video_processing_thread.start()

# Hàm để tạo stream video cho client
def generate_frames():
    global processed_frame_global, is_direction_switching, current_direction
    
    while True:
        # Lấy frame đã xử lý từ biến toàn cục
        with frame_lock:
            if processed_frame_global is not None:
                frame = processed_frame_global.copy()
            elif frame_global is not None:
                frame = frame_global.copy()
            else:
                # Tạo frame trống nếu chưa có frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for video...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Nếu đang chuyển hướng, hiển thị thông báo
        if is_direction_switching:
            cv2.putText(frame, f"Switching camera to {current_direction}...", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Thêm thông tin về đèn giao thông
        with traffic_light_lock:
            h_state = traffic_light_state["horizontal"]
            v_state = traffic_light_state["vertical"]
            h_time = traffic_light_state["horizontal_time"]
            v_time = traffic_light_state["vertical_time"]
        
        # Vẽ trạng thái đèn giao thông
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 120), (300, frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Hiển thị trạng thái đèn ngang
        h_color = (0, 0, 255) if h_state == "red" else (0, 255, 255) if h_state == "yellow" else (0, 255, 0)
        cv2.putText(frame, f"Horizontal: {h_state.upper()} ({h_time}s)", 
                   (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, h_color, 2)
        
        # Hiển thị trạng thái đèn dọc
        v_color = (0, 0, 255) if v_state == "red" else (0, 255, 255) if v_state == "yellow" else (0, 255, 0)
        cv2.putText(frame, f"Vertical: {v_state.upper()} ({v_time}s)", 
                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
        
        # Thêm thông tin timestamp
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Điều chỉnh kích thước frame để giảm độ trễ khi stream
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Chuyển đổi frame thành JPEG với chất lượng thấp hơn để giảm kích thước
        ret, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        # Tạo response
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Ngủ một chút để giảm tải CPU và mạng
        time.sleep(0.05)  # 20 fps, đủ cho streaming mượt mà mà không quá nặng

# Routes
@app.get("/")
async def root():
    return {"message": "Traffic Analysis System API"}

# Video streaming endpoint
@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(generate_frames(),
                           media_type='multipart/x-mixed-replace; boundary=frame')

# Endpoint để lấy thống kê giao thông
@app.get('/traffic_stats')
async def get_traffic_stats():
    with stats_lock:
        return {
            "horizontal_count": traffic_stats["horizontal_count"],
            "vertical_count": traffic_stats["vertical_count"],
            "horizontal_density": traffic_stats["horizontal_density"],
            "vertical_density": traffic_stats["vertical_density"],
            "current_direction": current_direction,
            "timestamp": traffic_stats["timestamp"]
        }

# Endpoint để lấy dữ liệu lịch sử
@app.get('/historical_data')
async def get_historical_data():
    return historical_data

# Endpoint để lấy trạng thái đèn giao thông
@app.get('/traffic_light_state')
async def get_traffic_light_state():
    with traffic_light_lock:
        return traffic_light_state

# Endpoint để thay đổi hướng camera
@app.post('/camera_direction')
async def set_camera_direction(direction_data: CameraDirection):
    global is_direction_switching
    
    direction = direction_data.direction
    if direction not in ["horizontal", "vertical"]:
        raise HTTPException(status_code=400, detail="Hướng không hợp lệ. Chỉ chấp nhận 'horizontal' hoặc 'vertical'")
    
    # Đánh dấu đang chuyển hướng
    is_direction_switching = True
    
    # Tạo một luồng để thực hiện chuyển hướng không đồng bộ
    def switch_direction_async():
        global is_direction_switching
        try:
            if ptz_controller:
                ptz_controller.move_to_direction(direction)
            else:
                # Fallback nếu không có kết nối PTZ
                global current_direction
                current_direction = direction
                time.sleep(2)  # Giả lập thời gian di chuyển
        except Exception as e:
            logger.error(f"Lỗi khi chuyển hướng camera: {e}")
        finally:
            is_direction_switching = False
    
    # Bắt đầu thread mới
    threading.Thread(target=switch_direction_async).start()
    
    return {"message": f"Đang chuyển camera sang hướng {direction}"}

# Endpoint để bật/tắt chế độ điều khiển thông minh
@app.post('/smart_mode/{enabled}')
async def set_smart_mode(enabled: bool):
    if traffic_light_controller:
        traffic_light_controller.set_smart_mode(enabled)
        return {"message": f"Chế độ thông minh: {'Bật' if enabled else 'Tắt'}"}
    else:
        raise HTTPException(status_code=500, detail="Bộ điều khiển đèn giao thông không khả dụng")

# Endpoint để gửi dữ liệu đến ThingSpeak
@app.post("/send-data/")
async def send_data_to_thingspeak(data: SensorData):
    try:
        # Tạo payload để gửi đến ThingSpeak
        payload = {
            'api_key': THINGSPEAK_WRITE_API_KEY,
            'field1': int(data.green),  # field1: green
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

# Endpoint để thiết lập lại trạng thái đèn giao thông
@app.post("/reset-traffic-light")
async def reset_traffic_light():
    global traffic_light_controller, traffic_light_state
    
    try:
        if traffic_light_controller:
            # Tạo lại bộ điều khiển đèn giao thông
            traffic_light_controller = TrafficLightController()
            
            # Cập nhật trạng thái
            with traffic_light_lock:
                traffic_light_state = traffic_light_controller.current_state.copy()
            
            return {"message": "Đã thiết lập lại trạng thái đèn giao thông"}
        else:
            raise HTTPException(status_code=500, detail="Bộ điều khiển đèn giao thông không khả dụng")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

# Endpoint để lấy cấu hình hệ thống
@app.get("/system-config")
async def get_system_config():
    return {
        "camera": {
            "ip": CAMERA_IP,
            "rtsp_port": RTSP_PORT,
            "onvif_port": ONVIF_PORT,
            "direction_switch_interval": Config.DIRECTION_SWITCH_INTERVAL
        },
        "detection": {
            "model": Config.DEFAULT_MODEL,
            "inference_size": Config.INFERENCE_SIZE,
            "detection_interval": Config.DETECTION_INTERVAL,
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD
        },
        "traffic_light": {
            "min_green_time": Config.MIN_GREEN_TIME,
            "max_green_time": Config.MAX_GREEN_TIME,
            "yellow_time": Config.YELLOW_TIME,
            "smart_mode": traffic_light_controller.smart_mode_enabled if traffic_light_controller else True
        }
    }

# Khởi động ứng dụng
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)