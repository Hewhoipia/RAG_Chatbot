# Redis/Elasticsearch trong môi trường Offline (Linux/Ubuntu)

* **Cách 1: Cài đặt trực tiếp bằng gói (Package Installation)**
* Dùng máy có mạng tải file `.deb` (cho Ubuntu/Debian) hoặc `.rpm` (cho CentOS/RHEL) của Elasticsearch hoặc Redis Stack.
* Copy file cài đặt vào server offline và chạy lệnh: `sudo dpkg -i <tên_file.deb>`.


* **Cách 2: Sử dụng Docker (Khuyên dùng vì tính đồng nhất)**
* Trên máy có mạng, kéo image về: `docker pull redis/redis-stack:latest`
* Lưu image thành file nén: `docker save redis/redis-stack:latest -o redis_stack.tar`
* Copy file `redis_stack.tar` vào server offline.
* Load image lên server offline: `docker load -i redis_stack.tar`
* Chạy container. Mọi truy xuất dữ liệu từ FastAPI đến database đều diễn ra qua cổng `localhost` (ví dụ 6379 cho Redis, 9200 cho ES), hoàn toàn khép kín.

Nên chọn **Redis Stack**. Nó chạy hoàn toàn trên RAM (In-memory) và tích hợp sẵn lệnh `FT.SUGADD` và `FT.SUGGET` được thiết kế tối ưu bằng cấu trúc Trie ở cấp độ ngôn ngữ C, cho tốc độ tính bằng micro-giây. Elasticsearch mạnh nhưng khá nặng nề và tiêu tốn nhiều RAM của server hệ thống.