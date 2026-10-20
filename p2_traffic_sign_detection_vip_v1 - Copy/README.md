# Điều khiển bám làn đã cân chỉnh

Chạy từ thư mục gốc dự án (dùng đúng môi trường Python của simulator):

```powershell
python -m pip install -r p2_traffic_sign_detection_vip_v1/requirements.txt
python p2_traffic_sign_detection_vip_v1/drive.py
```

Server nhận ảnh tại cổng 4567. Đường dẫn model được tính theo vị trí `drive.py`.

## Thay đổi chính

- Dò 3 dải ở 68%, 76%, 84% chiều cao ảnh; đã bỏ hai dải xa nhất ở 54% và 60%. Số nhỏ hơn tương ứng nhìn xa hơn. Ưu tiên vạch trắng, dùng biên ảnh làm dự phòng và kiểm tra độ liền mạch giữa các dải.
- Ước lượng bề rộng làn riêng cho từng khoảng nhìn thay vì cố định 400 pixel. Khi chỉ có một vạch, dùng bề rộng đã học và hạ độ tin cậy.
- Sai số lái được chuẩn hóa theo kích thước ảnh. Lọc sai số, lọc đạo hàm, có vùng chết quanh tâm và giới hạn tốc độ đổi góc lái theo thời gian.
- Ga đường thẳng tối đa 0.58, giảm khi cua hoặc độ tin cậy thấp. Mất làn thì hạ ga; sau 0.75 giây không nhận diện đáng tin cậy, ga về 0. Ga 0 là ngừng cấp ga theo giao thức hiện tại, không có lệnh phanh riêng.
- Lệnh rẽ từ biển báo cũng qua giới hạn đổi góc. Biển đi thẳng không còn ép ga tối đa. Biển đã xác nhận không bị liên tục dời lịch hoặc thực hiện lại khi còn xuất hiện liên tục.
- Đếm chung biển rẽ trái/phải đã xác nhận trong mỗi kết nối simulator. Riêng biển rẽ thứ 3 chờ thêm 0.4 giây: tổng thời gian từ xác nhận đến bắt đầu lệnh là 0.7 giây (0.3 + 0.4). Biển dừng, đi thẳng, cấm rẽ không tăng bộ đếm. Trong lúc chờ, xe vẫn bám làn bình thường. Kết nối lại simulator sẽ đặt bộ đếm về 0. Chỉnh phần chờ thêm bằng `THIRD_TURN_EXTRA_DELAY_SECONDS` trong `drive.py`.
- Ưu tiên ảnh mới cho tiến trình biển báo; bỏ kết quả từ ảnh cũ hơn 0.5 giây.

## Cân chỉnh sau khi chạy simulator

Các hằng số ở đầu `lane_line_detection.py`:

| Tham số | Mặc định | Cách chỉnh |
| --- | --- | --- |
| `CRUISE_THROTTLE` | 0.58 | Nếu còn đảo lái ở tốc độ cao, thử 0.48–0.52 trước. |
| `STEERING_KP` | 1.05 | Nếu ôm cua thiếu, tăng từng 0.10; nếu lắc, giảm từng 0.10. |
| `STEERING_RATE` | 1.6 / giây | Nếu đổi hướng chậm, thử 1.8; nếu giật, thử 1.3–1.4. |
| `MAX_STEERING` | 0.75 | Giới hạn góc tối đa cho cả bám làn và biển báo. |
| `SCAN_ROWS` | 0.84 → 0.68 | Vùng nhìn hiện tại. Khi đổi camera, chỉnh cùng `INITIAL_WIDTHS`. |

Chấm đỏ trong cửa sổ Result là tâm làn từng dải; vạch xanh là tín hiệu tốt hơn, vàng là ước lượng độ tin cậy thấp. `Lane confidence` cho biết độ hỗ trợ nhận diện, không phải xác suất đã hiệu chuẩn.

`current_speed` chưa có đơn vị xác định trong dự án nên không dùng công thức trừ trực tiếp khỏi ga. Tốc độ hiện được điều tiết bằng độ cua và độ tin cậy làn. Lệnh biển báo vẫn dùng thời lượng cố định trong `DURATION_BY_SIGN`; cần kiểm tra trên bản đồ thực tế khi thay ga/camera.

## Kiểm tra

```powershell
python -m unittest discover -s p2_traffic_sign_detection_vip_v1 -p 'test_*.py' -v
```

Bộ kiểm tra gồm đường tổng hợp ở nhiều độ phân giải, cua ở phía xa, nhiễu, một vạch, mất làn, đổi hướng, biển báo lặp và truyền ảnh qua WebSocket thật trên máy. Đồng thời kiểm tra 10 ảnh trong `content/images/lane_line_images.zip`. Các kiểm tra này chưa thay thế lượt chạy vòng kín trên simulator.
