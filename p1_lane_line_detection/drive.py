import asyncio
import websockets
from PIL import Image
import json
import cv2
import numpy as np
import base64
from io import BytesIO

# import calculate_control_signal and birdview_transform if available
try:
    from lane_line_detection import calculate_control_signal, birdview_transform
except Exception:
    try:
        from lane_line_detection import calculate_control_signal
    except Exception:
        calculate_control_signal = None
    birdview_transform = None


async def echo(websocket, path):
    async for message in websocket:
        # Giải mã ảnh từ simulator
        try:
            data = json.loads(message)
        except Exception:
            print("Invalid JSON from simulator, skipping.")
            continue

        b64 = data.get("image")
        if not b64:
            print("No image field, skipping.")
            continue

        try:
            pil_img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            image = np.asarray(pil_img)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print("Failed to decode image:", e)
            continue

        # Chuẩn bị ảnh để vẽ kết quả
        draw = image.copy()

        # Nếu có birdview_transform, hiển thị birdview
        if birdview_transform is not None:
            try:
                bird_img = birdview_transform(image)
                # tùy chọn resize nếu quá lớn:
                # bird_disp = cv2.resize(bird_img, (int(bird_img.shape[1]*0.7), int(bird_img.shape[0]*0.7)))
                cv2.imshow("Birdview", bird_img)
                cv2.waitKey(1)
            except Exception as e:
                # không crash nếu birdview lỗi
                # print("Birdview show error:", e)
                pass

        # Tính toán điều khiển
        throttle, steering_angle = 0.0, 0.0
        if calculate_control_signal is None:
            print("calculate_control_signal not available.")
        else:
            try:
                # gọi theo kiểu image, draw=draw
                throttle, steering_angle = calculate_control_signal(image, draw=draw)
            except TypeError:
                # fallback: try (current_speed, image) if simulator cung cấp current_speed
                cs = data.get("current_speed", 0.0)
                try:
                    throttle, steering_angle = calculate_control_signal(cs, image)
                except Exception as e:
                    print("calculate_control_signal failed fallback:", e)
                    throttle, steering_angle = 0.0, 0.0
            except Exception as e:
                print("calculate_control_signal runtime error:", e)
                throttle, steering_angle = 0.0, 0.0

        # Hiển thị kết quả
        try:
            cv2.imshow("Result", draw)
            cv2.waitKey(1)
        except Exception:
            pass

        # Gửi kết quả lại simulator
        send_msg = json.dumps({"throttle": float(throttle), "steering": float(steering_angle)})
        print("SEND:", send_msg)
        try:
            await websocket.send(send_msg)
        except Exception as e:
            print("Failed to send message:", e)


async def main():
    async with websockets.serve(echo, "0.0.0.0", 4567, ping_interval=None):
        print("Server started on 0.0.0.0:4567")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        cv2.destroyAllWindows()