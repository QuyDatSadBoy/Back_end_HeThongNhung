import cv2
import time
from onvif import ONVIFCamera
import threading
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("camera_control.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CameraControl")

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
        
        # Hướng hiện tại
        self.current_direction = "horizontal"
        
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
            self.camera = ONVIFCamera(self.ip, self.port, self.username, self.password, wsdl_dir='/etc/onvif/wsdl/')
            
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
            if preset_id == 1:
                self.current_direction = "horizontal"
            elif preset_id == 2:
                self.current_direction = "vertical"
                
            return True
        except Exception as e:
            logger.error(f"Lỗi di chuyển đến preset: {e}")
            return False

    def move_to_direction(self, direction):
        """Di chuyển camera đến hướng cụ thể"""
        if direction not in ["horizontal", "vertical"]:
            logger.error(f"Hướng không hợp lệ: {direction}")
            return False
            
        # Nếu đã ở hướng cần chuyển, không cần làm gì
        if direction == self.current_direction:
            logger.info(f"Camera đã ở hướng {direction}")
            return True
            
        logger.info(f"Đang chuyển camera từ {self.current_direction} sang {direction}")
        
        if not self.connected:
            # Chế độ mô phỏng
            logger.info(f"Mô phỏng: Chuyển hướng sang {direction}")
            self.current_direction = direction
            time.sleep(2)  # Giả lập thời gian di chuyển
            return True
            
        # Di chuyển đến preset tương ứng
        preset_id = 1 if direction == "horizontal" else 2
        success = self.goto_preset(preset_id)
        
        if success:
            time.sleep(2)  # Đợi camera di chuyển hoàn tất
            
        return success

    def get_current_direction(self):
        """Trả về hướng hiện tại của camera"""
        return self.current_direction

# Hàm để stream video từ camera với độ trễ thấp
def stream_camera(rtsp_url, ptz_controller, switch_interval=30):
    """
    Stream video từ camera và xoay camera theo định kỳ
    
    Args:
        rtsp_url (str): Đường dẫn RTSP đến camera
        ptz_controller: Bộ điều khiển PTZ
        switch_interval (int): Thời gian (giây) giữa các lần chuyển hướng
    """
    # Cấu hình OpenCV để giảm độ trễ
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024000|max_delay;0"
    
    # Mở kết nối camera
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer nhỏ để giảm độ trễ
    
    if not cap.isOpened():
        logger.error(f"Không thể kết nối đến camera tại {rtsp_url}")
        return
    
    logger.info(f"Đã kết nối thành công đến camera tại {rtsp_url}")
    
    # Lấy thông tin kích thước frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    logger.info(f"Độ phân giải: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Thời điểm chuyển hướng lần cuối
    last_switch_time = time.time()
    
    # Flag để theo dõi trạng thái chuyển hướng
    is_switching = False
    
    try:
        while True:
            # Đọc frame từ camera
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Không thể đọc frame từ camera. Đang thử kết nối lại...")
                time.sleep(1)
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                continue
            
            # Kiểm tra xem có cần chuyển hướng không
            current_time = time.time()
            if current_time - last_switch_time >= switch_interval and not is_switching:
                is_switching = True
                new_direction = "vertical" if ptz_controller.get_current_direction() == "horizontal" else "horizontal"
                
                # Tạo thread mới để xoay camera (không chặn luồng chính)
                def switch_direction():
                    nonlocal is_switching, last_switch_time
                    logger.info(f"Đang chuyển camera sang hướng {new_direction}")
                    ptz_controller.move_to_direction(new_direction)
                    last_switch_time = time.time()
                    is_switching = False
                    logger.info(f"Đã chuyển xong camera sang hướng {new_direction}")
                
                threading.Thread(target=switch_direction).start()
            
            # Hiển thị thông tin trên frame
            current_direction = ptz_controller.get_current_direction()
            cv2.putText(
                frame,
                f"Direction: {current_direction}" + (" (switching...)" if is_switching else ""),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )
            
            # Hiển thị frame
            cv2.imshow("Camera Stream", frame)
            
            # Nhấn 'q' để thoát
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                # Chuyển sang hướng ngang
                threading.Thread(target=lambda: ptz_controller.move_to_direction("horizontal")).start()
            elif key == ord('v'):
                # Chuyển sang hướng dọc
                threading.Thread(target=lambda: ptz_controller.move_to_direction("vertical")).start()
    
    except KeyboardInterrupt:
        logger.info("Đã dừng stream camera")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Điều khiển camera IMOU với 2 hướng cố định')
    
    parser.add_argument('--ip', type=str, default='192.168.43.81',
                        help='Địa chỉ IP của camera')
    parser.add_argument('--rtsp-port', type=int, default=554,
                        help='Cổng RTSP (thường là 554)')
    parser.add_argument('--onvif-port', type=int, default=80,
                        help='Cổng ONVIF (thường là 80)')
    parser.add_argument('--username', type=str, default='admin',
                        help='Tên đăng nhập camera')
    parser.add_argument('--password', type=str, default='L2A17879',
                        help='Mật khẩu camera')
    parser.add_argument('--channel', type=int, default=1,
                        help='Kênh camera (thường là 1)')
    parser.add_argument('--stream', type=int, default=1,
                        help='Loại stream (0=main stream, 1=sub stream)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Thời gian (giây) giữa các lần chuyển hướng camera')
    
    args = parser.parse_args()
    
    # Tạo URL RTSP
    rtsp_url = f"rtsp://{args.username}:{args.password}@{args.ip}:{args.rtsp_port}/cam/realmonitor?channel={args.channel}&subtype={args.stream}"
    
    # Khởi tạo bộ điều khiển PTZ
    ptz_controller = ImouPTZController(args.ip, args.onvif_port, args.username, args.password)
    
    # Bắt đầu stream camera
    stream_camera(rtsp_url, ptz_controller, args.interval)