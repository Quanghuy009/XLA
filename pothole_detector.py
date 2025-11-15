# pothole_detector.py
import os
from pathlib import Path
from utils import predict_single_image, predict_video

# Đường dẫn mô hình
MODEL_PATH = Path("train_optimized3/weights/best.pt")

if not MODEL_PATH.exists():
    print("Không tìm thấy mô hình! Hãy kiểm tra lại đường dẫn:")
    print("   ", MODEL_PATH)
    exit()

print("Pothole Detection Demo - Nhận diện ổ gà trên đường")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print("Chọn chế độ:")
print("   1. Test trên ảnh (hiển thị đẹp, lưu ảnh kết quả)")
print("   2. Test trên video (realtime + lưu video output)")
print("   3. Thoát")
print("-" * 60)

while True:
    choice = input("Nhập lựa chọn (1/2/3): ").strip()
    
    if choice == "1":
        print("\nDanh sách file ảnh có sẵn trong thư mục inputs/:")
        images = list(Path("inputs").glob("*.[jpg|png|jpeg]*"))
        if not images:
            print("   (Chưa có ảnh nào)")
        else:
            for i, img in enumerate(images[:10]):
                print(f"  {i+1}. {img.name}")
        
        print("\n→ Bạn có thể:")
        print("   • Nhấn Enter để chọn ảnh đầu tiên")
        print("   • Gõ số thứ tự (ví dụ: 6)")
        print("   • Hoặc dán đường dẫn đầy đủ")
        
        sel = input("   Nhập ở đây: ").strip()
        
        if sel == "" and images:
            path = str(images[0])
            print(f"Đang dùng ảnh đầu tiên: {images[0].name}")
        elif sel.isdigit() and 1 <= int(sel) <= len(images):
            idx = int(sel) - 1
            path = str(images[idx])
            print(f"Đã chọn: {images[idx].name}")
        else:
            path = sel or input("   Dán đường dẫn đầy đủ đến ảnh: ").strip()
        
        if not Path(path).exists():
            print("Không tìm thấy ảnh! Quay lại menu...\n")
            continue
            
        predict_single_image(
            image_path=path,
            model_path=str(MODEL_PATH),
            conf_threshold=0.1,
            save_output=True
        )

    elif choice == "2":
        print("\nDanh sách video có sẵn trong inputs/ (hỗ trợ mọi loại MP4):")
        videos = list(Path("inputs").glob("*.mp4"))
        if not videos:
            print("   (Chưa có video nào – bạn có thể dán đường dẫn bên dưới)")
        else:
            for i, vid in enumerate(videos[:10]):
                print(f"  {i+1}. {vid.name}")
        
        print("\n→ Bạn có thể:")
        print("   • Nhấn Enter để chọn video đầu tiên")
        print("   • Gõ số thứ tự (ví dụ: 2)")
        print("   • Hoặc dán đường dẫn đầy đủ đến file .mp4")
        
        sel = input("   Nhập ở đây: ").strip()
        
        if sel == "" and videos:
            path = str(videos[0])
            print(f"Đang dùng video đầu tiên: {videos[0].name}")
        elif sel.isdigit() and 1 <= int(sel) <= len(videos):
            idx = int(sel) - 1
            path = str(videos[idx])
            print(f"Đã chọn: {videos[idx].name}")
        else:
            path = sel or input("   Dán đường dẫn đầy đủ đến video MP4: ").strip()
        
        if not Path(path).exists():
            print("Không tìm thấy video! Quay lại menu...\n")
            continue
        
        print("Đang mở video, vui lòng đợi vài giây...")
        predict_video(
            video_path=path,
            model_path=str(MODEL_PATH),
            conf_threshold=0.05,
            save_output=True
        )

    elif choice == "3":
        break
    else:
        print("Lựa chọn không hợp lệ, vui lòng nhập lại!\n")