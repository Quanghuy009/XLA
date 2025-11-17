# pothole_gui_final_result_only.py
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import queue
import time

# ====================== CẤU HÌNH ======================
MODEL_PATH = Path("train_with_newdataset/weights/best.pt")
INPUT_DIR = Path("inputs")
OUTPUT_DIR = Path("outputs")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# NGƯỠNG TIN CẬY 
CONF_THRESHOLD = 0.3   

# Tối ưu GUI
DISPLAY_FPS = 30
FRAME_DELAY_MS = int(1000 / DISPLAY_FPS)

if not MODEL_PATH.exists():
    messagebox.showerror("LỖI", f"Không tìm thấy mô hình:\n{MODEL_PATH}")
    exit()

model = YOLO(str(MODEL_PATH))

# ====================== HÀM XỬ LÝ ======================
def predict_single_image(image_path, conf_threshold, save_output=True):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError("Không tìm thấy ảnh!")
    
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)[0]
    annotated_img = results.plot()  # Chỉ lấy ảnh có box
    
    num = len(results.boxes) if results.boxes is not None else 0
    
    if save_output and num > 0:
        output_path = OUTPUT_DIR / f"{image_path.stem}_pothole_detected.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        return annotated_img, num, str(output_path)
    
    return annotated_img, num, None


def predict_video_realtime(video_path, conf_threshold, save_output=True, frame_queue=None, stop_event=None):
    cap = None
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_FFMPEG]
    for api in backends:
        cap = cv2.VideoCapture(video_path, api)
        if cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        raise ConnectionError("Không mở được video!")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = OUTPUT_DIR / f"{Path(video_path).stem}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    frame_count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
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
        
        if frame_queue and frame_queue.qsize() < 2:
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_queue.put(frame_rgb)
        
        out.write(annotated)
    
    cap.release()
    out.release()
    return str(output_path)


# ====================== GUI CLASS ======================
class PotholeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pothole Detection")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f2f5")

        self.photo = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.video_thread = None
        self.stop_event = threading.Event()

        self.setup_ui()
        self.root.after(FRAME_DELAY_MS, self.update_frame)

    def setup_ui(self):
        tk.Label(self.root, text="NHẬN DIỆN Ổ GÀ THÔNG MINH", font=("Helvetica", 22, "bold"),
                 bg="#f0f2f5", fg="#2c3e50").pack(pady=15)

        main = tk.Frame(self.root, bg="#f0f2f5")
        main.pack(fill="both", expand=True, padx=20)

        # === Trái: Điều khiển ===
        left = tk.Frame(main, width=400, bg="white", relief="groove", bd=3)
        left.pack(side="left", fill="y", padx=(0,15))
        left.pack_propagate(False)

        tk.Label(left, text="ĐIỀU KHIỂN", font=("Arial", 14, "bold"), bg="white").pack(pady=12)
        tk.Button(left, text="CHỌN ẢNH", command=self.select_image, width=25, height=2, bg="#3498db", fg="white").pack(pady=8)
        tk.Button(left, text="CHỌN VIDEO", command=self.select_video, width=25, height=2, bg="#e74c3c", fg="white").pack(pady=8)
        tk.Button(left, text="DỪNG VIDEO", command=self.stop_video, width=25, height=1, bg="#e67e22", fg="white").pack(pady=8)

        self.status = tk.Label(left, text="Sẵn sàng", bg="white", fg="#7f8c8d", anchor="w", padx=15, wraplength=350)
        self.status.pack(fill="x", pady=10)

        # === Phải: Hiển thị kết quả ===
        right = tk.Frame(main, bg="white", relief="groove", bd=3)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="KẾT QUẢ NHẬN DIỆN", font=("Arial", 14, "bold"), bg="white").pack(pady=10)
        self.result_label = tk.Label(right, text="Chọn file để xử lý", bg="white", fg="#7f8c8d", font=("Arial", 12))
        self.result_label.pack(pady=5)

        # Canvas n
        self.canvas = tk.Canvas(right, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=20, pady=10)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Hình ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path: 
            self.run_image(path)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4")])
        if path: 
            self.run_video(path)

    def run_image(self, path):
        self.stop_video()
        self.status.config(text="Đang xử lý ảnh...")
        self.result_label.config(text="Đang nhận diện...")
        threading.Thread(target=self.process_image, args=(path,), daemon=True).start()

    def process_image(self, path):
        try:
            annotated, count, saved = predict_single_image(path, CONF_THRESHOLD, True)
            self.root.after(0, lambda: self.show_result_image(annotated, count, saved))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi", str(e)))
        finally:
            self.root.after(0, lambda: self.status.config(text="Sẵn sàng"))

    def show_result_image(self, annotated_img, count, saved):
        self.display_frame(annotated_img)
        msg = f"Phát hiện {count} ổ gà"
        self.result_label.config(text=msg, fg="#27ae60")

    def run_video(self, path):
        self.stop_video()
        self.stop_event.clear()
        self.status.config(text="Đang xử lý video...")
        self.result_label.config(text="Đang phát video...")
        
        self.video_thread = threading.Thread(
            target=predict_video_realtime,
            args=(path, CONF_THRESHOLD, True, self.frame_queue, self.stop_event),
            daemon=True
        )
        self.video_thread.start()

    def update_frame(self):
        if not self.frame_queue.empty():
            try:
                # Bỏ frame cũ nếu có nhiều
                while self.frame_queue.qsize() > 1:
                    self.frame_queue.get_nowait()
                frame = self.frame_queue.get_nowait()
                self.display_frame(frame)
            except:
                pass
        self.root.after(FRAME_DELAY_MS, self.update_frame)

    # HÀM HIỂN THỊ MỚI – CHỈ KẾT QUẢ, FIT VỪA KHUNG, GIỮ TỶ LỆ
    def display_frame(self, frame_rgb):
        img = Image.fromarray(frame_rgb)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 1000, 600  # Giá trị mặc định khi chưa load

        # Tính tỷ lệ để fit vừa khung (giữ nguyên tỷ lệ ảnh)
        img_ratio = img.width / img.height
        canvas_ratio = canvas_w / canvas_h

        if img_ratio > canvas_ratio:
            new_w = canvas_w
            new_h = int(canvas_w / img_ratio)
        else:
            new_h = canvas_h
            new_w = int(canvas_h * img_ratio)

        # Resize mượt mà
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)

        # Xóa cũ, vẽ mới vào giữa
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=self.photo)

    def stop_video(self):
        self.stop_event.set()
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait()
            except: pass
        self.canvas.delete("all")
        self.result_label.config(text="Đã dừng video", fg="#e67e22")

    def __del__(self):
        self.stop_video()


# ====================== CHẠY ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_video(), root.destroy()))
    root.mainloop()