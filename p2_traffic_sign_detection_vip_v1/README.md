# Điều khiển bám làn đã cân chỉnh

Chạy từ thư mục gốc dự án (dùng đúng môi trường Python của simulator):

```powershell
python -m pip install -r p2_traffic_sign_detection_vip_v1/requirements.txt
python p2_traffic_sign_detection_vip_v1/drive.py
```

Server nhận ảnh tại cổng 4567. Đường dẫn model được tính theo vị trí `drive.py`.

## Thay đổi chính

- Dò 3 dải bám làn ở 68%, 76%, 84% chiều cao ảnh và thêm dải báo mất làn phía xa ở 64%, xa hơn dải 68% một chút. Dải 64% chỉ báo mất làn sớm, không tham gia tính góc bám làn thông thường. Đã bỏ hai dải cũ ở 54% và 60%. Số nhỏ hơn tương ứng nhìn xa hơn. Ưu tiên vạch trắng, dùng biên ảnh làm dự phòng và kiểm tra độ liền mạch giữa các dải.
- Ước lượng bề rộng làn riêng cho từng khoảng nhìn thay vì cố định 400 pixel. Khi chỉ có một vạch, dùng bề rộng đã học và hạ độ tin cậy.
- Sai số lái được chuẩn hóa theo kích thước ảnh. Lọc sai số, lọc đạo hàm, có vùng chết quanh tâm và giới hạn tốc độ đổi góc lái theo thời gian.
- Ga giảm khi cua hoặc độ tin cậy thấp. Mất làn thì hạ ga; sau 0.75 giây không nhận diện đáng tin cậy, ga bám làn về 0. Nếu đang thực hiện lệnh rẽ tại ngã ba, xe dùng ga riêng cho lệnh rẽ trong thời lượng rẽ để đi qua vùng không có vạch. Ga 0 là ngừng cấp ga theo giao thức hiện tại, không có lệnh phanh riêng.
- Lệnh rẽ từ biển báo cũng qua giới hạn đổi góc. Biển đi thẳng không còn ép ga tối đa. Biển đã xác nhận không bị liên tục dời lịch hoặc thực hiện lại khi còn xuất hiện liên tục.
- Biển rẽ trái/phải được ghi nhớ, không tự rẽ theo thời gian kể từ khi thấy biển. Xe tiếp tục bám làn cho tới khi mất làn liên tục ít nhất 0.12 giây; lúc đó mới bắt đầu lệnh rẽ đã ghi nhớ. Đây là cách dùng mất làn làm dấu hiệu ngã ba, chưa phải bộ nhận diện hình học ngã ba riêng. Chỉ ở lần mất làn không có biển thứ 2 trong mỗi kết nối simulator, xe mới bắt đầu rẽ phải ngay với góc đầu ra bằng `FALLBACK_RIGHT_STEERING = +1.0` (toàn bộ hành trình lái theo giao thức), không chờ thêm 0.12 giây. Riêng nhánh dự phòng này bỏ giới hạn tốc độ đổi lái để xuất góc tối đa ngay trong khung hình kích hoạt; bám làn và rẽ theo biển vẫn giữ làm mượt. Nhánh dự phòng dùng ga rẽ phải và thời lượng `DURATION_BY_SIGN['right']` hiện có; áp dụng cho mọi lần mất làn không có biển, không giới hạn riêng map 1.
- Dải 64% phải thấy làn ít nhất một lần trước khi báo mất làn phía xa. Dải này chỉ tham gia đếm mất làn và kích hoạt rẽ phải dự phòng khi không có biển rẽ đang chờ. Khi đã ghi nhớ biển rẽ, xe bỏ qua báo mất của dải 64% và chỉ rẽ sau khi ba dải gần (68%, 76%, 84%) mất làn liên tục đủ 0.12 giây. Đường tím là dải báo xa, đổi đỏ khi mất tín hiệu; `Far preview` hiển thị trạng thái.
- Các biển rẽ đã ghi nhớ dùng cùng điều kiện mất làn liên tục 0.12 giây; không có thời gian chờ thêm cho biển thứ 3. Biển dừng, đi thẳng, cấm rẽ vẫn dùng lịch thực hiện riêng.
- Lần mất làn không có biển đầu tiên chưa tự rẽ phải; xe vẫn dùng điều khiển bám làn/mất làn bình thường. Phải thấy lại làn rồi mất lần thứ 2 mới xuất góc phải +1.0 ngay. Nhiều khung hình mất liên tục chỉ tính một lần; mất dải xa rồi mất các dải gần trong cùng đợt vẫn chỉ tính một lần. Bộ đếm đặt về 0 khi simulator kết nối lại; các lần có biển rẽ chờ không tăng bộ đếm này. Nhánh rẽ phải dự phòng chỉ kích hoạt ở lần thứ 2, không tự kích hoạt tiếp ở lần thứ 3 trở đi. `Loss` trong cửa sổ Result hiển thị bộ đếm.
- Mỗi lần mất làn chỉ kích hoạt một lệnh rẽ. Sau khi lệnh kết thúc, xe phải thấy lại làn và dải báo xa không còn báo mất mới được kích hoạt ở lần mất làn tiếp theo. Biển đang thực hiện hoặc chờ thực hiện (như dừng, cấm rẽ phải) được ưu tiên hơn nhánh dự phòng.
- Biển rẽ chờ hết hạn sau 15 giây nếu chưa có ngã ba để tránh áp dụng biển cũ cho ngã rẽ quá xa. Cửa sổ Result hiển thị `Wait` là hướng đang chờ và `Active` là lệnh đang thực hiện.
- Ưu tiên ảnh mới cho tiến trình biển báo; bỏ kết quả từ ảnh cũ hơn 0.5 giây.

## Cân chỉnh sau khi chạy simulator

Các hằng số ở đầu `lane_line_detection.py`:

| Tham số | Mặc định | Cách chỉnh |
| --- | --- | --- |
| `CRUISE_THROTTLE` | 0.58 | Nếu còn đảo lái ở tốc độ cao, thử 0.48–0.52 trước. |
| `STEERING_KP` | 1.05 | Nếu ôm cua thiếu, tăng từng 0.10; nếu lắc, giảm từng 0.10. |
| `STEERING_RATE` | 1.6 / giây | Nếu đổi hướng chậm, thử 1.8; nếu giật, thử 1.3–1.4. |
| `MAX_STEERING` | 0.75 | Giới hạn góc tối đa cho cả bám làn và biển báo. |
| `SCAN_ROWS` | 0.84 → 0.64 | Vùng nhìn hiện tại. Khi đổi camera, chỉnh cùng `INITIAL_WIDTHS`. |

Chấm đỏ trong cửa sổ Result là tâm làn từng dải; vạch xanh là tín hiệu tốt hơn, vàng là ước lượng độ tin cậy thấp. `Lane confidence` cho biết độ hỗ trợ nhận diện, không phải xác suất đã hiệu chuẩn.

`current_speed` chưa có đơn vị xác định trong dự án nên không dùng công thức trừ trực tiếp khỏi ga. Tốc độ hiện được điều tiết bằng độ cua và độ tin cậy làn. Lệnh biển báo vẫn dùng thời lượng cố định trong `DURATION_BY_SIGN`; cần kiểm tra trên bản đồ thực tế khi thay ga/camera.

## Kiểm tra

```powershell
python -m unittest discover -s p2_traffic_sign_detection_vip_v1 -p 'test_*.py' -v
```

Bộ kiểm tra gồm đường tổng hợp ở nhiều độ phân giải, cua ở phía xa, nhiễu, một vạch, mất làn, đổi hướng, biển báo lặp và truyền ảnh qua WebSocket thật trên máy. Đồng thời kiểm tra 10 ảnh trong `content/images/lane_line_images.zip`. Các kiểm tra này chưa thay thế lượt chạy vòng kín trên simulator.
