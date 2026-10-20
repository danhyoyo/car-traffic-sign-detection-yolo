# drive.py
import asyncio
import websockets
from PIL import Image
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from lane_line_detection import calculate_control_signal

# helper: clamp number
def clamp(x, lo, hi):
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))

async def handle_message(websocket):
    """
    Xử lý từng kết nối websocket.
    """
    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print("Invalid JSON received, skip.")
            continue

        # Lấy ảnh base64
        b64img = data.get("image")
        if not b64img:
            print("No image field in message, skip.")
            continue

        # Nếu có current_speed từ simulator, dùng nó (không bắt buộc)
        current_speed = data.get("current_speed", None)

        # decode ảnh an toàn
        try:
            img_bytes = base64.b64decode(b64img)
            pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
            image = np.asarray(pil_img)
            # PIL -> numpy trả về RGB, OpenCV mặc định dùng BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Failed to decode/convert image:", e)
            continue

        # Prepare visualization image
        draw = image.copy()

        # Gọi calculate_control_signal: hỗ trợ 2 kiểu signature
        throttle = 0.0
        steering_angle = 0.0
        try:
            # Thử gọi theo signature (image, draw=...)
            # Một số module của bạn có thể là: calculate_control_signal(image, draw=draw)
            throttle, steering_angle = calculate_control_signal(image, draw=draw)
        except TypeError:
            try:
                # Một số khác có signature (current_speed, image)
                # nếu current_speed không có thì truyền 0.0
                sp = current_speed if current_speed is not None else 0.0
                throttle, steering_angle = calculate_control_signal(sp, image)
            except Exception as e:
                print("calculate_control_signal failed:", e)
                # fallback: giữ nguyên 0.0
                throttle, steering_angle = 0.0, 0.0
        except Exception as e:
            print("calculate_control_signal unexpected error:", e)
            throttle, steering_angle = 0.0, 0.0

        # đảm bảo kiểu dữ liệu và giới hạn an toàn
        throttle = clamp(throttle, 0.0, 1.0)   # giả sử throttle từ 0..1
        steering_angle = clamp(steering_angle, -1.0, 1.0)  # steering -1..1

        # Hiển thị kết quả (nếu muốn)
        try:
            # thêm text lên ảnh để debug
            text = f"T:{throttle:.3f} S:{steering_angle:.3f}"
            cv2.putText(draw, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Result", draw)
            # waitKey(1) non-blocking; nếu muốn dừng khi nhấn phím, xử lý ở chỗ khác
            cv2.waitKey(1)
        except Exception as e:
            print("OpenCV display error:", e)

        # Gửi lại JSON (chỉ gửi các giá trị cần thiết)
        send_msg = json.dumps({
            "throttle": float(throttle),
            "steering": float(steering_angle)
        })
        print("SEND:", send_msg)
        try:
            await websocket.send(send_msg)
        except Exception as e:
            print("Failed to send message:", e)
            # nếu gửi thất bại, vòng for sẽ tự tiếp tục hoặc đóng connection

async def echo(websocket, path):
    """
    wrapper: giữ compat với websockets.serve signature
    """
    try:
        await handle_message(websocket)
    except websockets.ConnectionClosed:
        # kết nối đóng bình thường
        pass
    except Exception as e:
        print("Connection handler error:", e)

async def main():
    # ping_interval=None để tránh timeouts nếu simulator không ping
    async with websockets.serve(echo, "0.0.0.0", 4567, ping_interval=None):
        print("WebSocket server started on 0.0.0.0:4567")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        cv2.destroyAllWindows()
