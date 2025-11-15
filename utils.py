# utils.py
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import time
import torch
torch.cuda.empty_cache() if torch.cuda.is_available() else None 

def predict_single_image(image_path, model_path, conf_threshold=0.25, save_output=True):
    model = YOLO(model_path)
    image_path = Path(image_path)
    
    if not image_path.exists():
        print("Không tìm thấy ảnh!")
        return
    
    print(f"Đang phân tích: {image_path.name}")
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)[0]
    
    orig_img = cv2.imread(str(image_path))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    annotated_img = results.plot()
    
    num = len(results.boxes) if results.boxes is not None else 0
    print(f"Phát hiện: {num} ổ gà")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    axes[0].imshow(orig_img)
    axes[0].set_title("Ảnh Gốc", fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(annotated_img)
    axes[1].set_title(f"Kết Quả - Phát hiện {num} ổ gà", fontsize=16, fontweight='bold',
                      color='green' if num > 0 else 'red')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if save_output and num > 0:
        output_path = Path("outputs") / f"{image_path.stem}_pothole_detected.jpg"
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        print(f"Đã lưu: {output_path}")

def predict_video(video_path, model_path, conf_threshold=0.25, save_output=True):
    model = YOLO(model_path)
    video_path = str(Path(video_path))
    
    print(f"Đang thử mở video: {Path(video_path).name}")
    
    # Thử tất cả backend để mở được mọi loại MP4
    cap = None
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG]
    for api in backends:
        cap = cv2.VideoCapture(video_path, api)
        if cap.isOpened():
            print(f"Mở thành công với backend {api}")
            break
    
    if not cap.isOpened():
        print("Không mở được video bằng mọi cách!")
        print("→ File có thể bị lỗi hoặc codec quá mới")
        print("→ Giải pháp: Mở bằng VLC → Convert sang MP4 (H.264)")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Video mở được nhưng không đọc được frame → file hỏng!")
        cap.release()
        return
    
    print(f"Video chạy tốt! Độ phân giải: {frame.shape[1]}x{frame.shape[0]}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = Path("outputs") / f"{Path(video_path).stem}_detected.mp4"
    output_path.parent.mkdir(exist_ok=True)
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    print("Bắt đầu nhận diện... (Nhấn Q để dừng)")
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        annotated = results.plot()
        
        frame_count += 1
        elapsed = time.time() - start_time
        curr_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {curr_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow("Pothole Detection - Nhấn Q để thoát", annotated)
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"HOÀN THÀNH! Video kết quả lưu tại:\n   {output_path}")