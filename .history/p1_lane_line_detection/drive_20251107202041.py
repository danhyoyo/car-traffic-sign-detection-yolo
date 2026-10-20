import asyncio
import websockets
from PIL import Image
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from lane_line_detection import calculate_control_signal


async def echo(websocket, path):
    async for message in websocket:
        # Giải mã ảnh từ simulator
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Chuẩn bị ảnh để vẽ kết quả
        draw = image.copy()

        # Tính toán điều khiển
        throttle, steering_angle = calculate_control_signal(image, draw=draw)

        # Hiển thị kết quả
        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Gửi kết quả lại simulator
        message = json.dumps({"throttle": throttle, "steering": steering_angle})
        print(message)
        await websocket.send(message)


async def main():
    async with websockets.serve(echo, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # Chạy mãi mãi


asyncio.run(main())
