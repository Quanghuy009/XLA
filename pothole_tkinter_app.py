# pothole_tkinter_app.py - Phiên bản cập nhật: ẩn số, fps tùy chỉnh trong code
import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import os

# ====================== CẤU HÌNH ======================
MODEL_PATH = Path("train_optimized3/weights/best.pt")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    messagebox.showerror("LỖI", f"Không tìm thấy mô hình:\n{MODEL_PATH}")
    exit()

model = YOLO(str(MODEL_PATH))

DISPLAY_FPS = 30  # FPS hiển thị video, thay đổi ở đây nếu muốn nhanh/chậm

# ====================== APP CLASS ======================
class PotholeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận Diện Ổ Gà Thông Minh - YOLOv8")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f4f6f9")

        self.is_playing = False
        self.photo = None

        # Tiêu đề
        tk.Label(root, text="NHẬN DIỆN Ổ GÀ THÔNG MINH", font=("Helvetica", 24, "bold"), 
                 bg="#f4f6f9", fg="#2c3e50").pack(pady=20)

        main_frame = tk.Frame(root, bg="#f4f6f9")
        main_frame.pack(fill="both", expand=True, padx=25, pady=10)

        # === Bên trái: Điều khiển ===
        left = tk.Frame(main_frame, width=420, bg="white", relief="groove", bd=3)
        left.pack(side="left", fill="y", padx=(0, 20))
        left.pack_propagate(False)

        tk.Label(left, text="CHỌN FILE ĐẦU VÀO", font=("Arial", 15, "bold"), bg="white", fg="#2c3e50").pack(pady=20)

        tk.Button(left, text="CHỌN ẢNH", width=22, height=2, bg="#3498db", fg="white", font=("Arial", 11, "bold"),
                  command=self.select_image).pack(pady=15)
        tk.Button(left, text="CHỌN VIDEO", width=22, height=2, bg="#e74c3c", fg="white", font=("Arial", 11, "bold"),
                  command=self.select_video).pack(pady=15)

        self.status = tk.Label(left, text="Sẵn sàng", bg="white", fg="#7f8c8d", font=("Arial", 11), wraplength=380)
        self.status.pack(pady=20)

        self.progress = ttk.Progressbar(left, mode='indeterminate', length=320)
        self.progress.pack(pady=20, padx=40)

        # === Bên phải: Hiển thị ===
        right = tk.Frame(main_frame, bg="white", relief="groove", bd=3)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="KẾT QUẢ NHẬN DIỆN", font=("Arial", 15, "bold"), bg="white", fg="#2c3e50").pack(pady=15)

        self.info = tk.Label(right, text="Chọn ảnh hoặc video để bắt đầu", bg="white", fg="#7f8c8d", font=("Arial", 13))
        self.info.pack(pady=10)

        # Canvas hiển thị ảnh/video
        self.canvas = tk.Canvas(right, bg="#ecf0f1", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=25, pady=15)

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Hình ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        self.stop_video()
        self.status.config(text=f"Đang xử lý: {Path(path).name}")
        self.info.config(text="Đang nhận diện ổ gà...")
        threading.Thread(target=self.process_image, args=(path,), daemon=True).start()

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if not path: return
        self.stop_video()
        self.status.config(text=f"Phát video: {Path(path).name}")
        self.info.config(text="Đang xử lý và phát video...")
        threading.Thread(target=self.process_video_realtime, args=(path,), daemon=True).start()

    # ẩn số bên cạnh box
    def draw_boxes_no_conf(self, frame, results):
        if results.boxes is not None:
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = results.names[int(cls)]  # chỉ tên class, không có số
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return frame

    def process_image(self, img_path):
        self.progress.start()
        try:
            results = model(img_path, conf=0.1, verbose=False)[0]
            annotated = self.draw_boxes_no_conf(cv2.imread(str(img_path)), results)

            output_path = OUTPUT_DIR / f"result_{Path(img_path).stem}_detected.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            self.display_image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               f"Phát hiện {len(results.boxes) if results.boxes is not None else 0} ổ gà")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Xử lý ảnh thất bại:\n{e}")
        finally:
            self.progress.stop()

    def process_video_realtime(self, video_path):
        self.progress.start()
        self.is_playing = True
        try:
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = OUTPUT_DIR / f"result_{Path(video_path).stem}_detected.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_path), fourcc, DISPLAY_FPS, (w, h))

            while self.is_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=0.05, verbose=False)[0]
                annotated = self.draw_boxes_no_conf(frame.copy(), results)
                out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                self.display_image(frame_rgb, f"Đang phát video...")

                self.root.update_idletasks()
                time.sleep(1 / DISPLAY_FPS)  # điều chỉnh tốc độ hiển thị

            cap.release()
            out.release()
            if self.is_playing:
                self.info.config(text=f"HOÀN TẤT!\nĐã lưu: {output_path.name}", fg="#27ae60", font=("Arial", 14, "bold"))

        except Exception as e:
            messagebox.showerror("Lỗi", f"Video lỗi:\n{e}")
        finally:
            self.progress.stop()
            self.is_playing = False

    def display_image(self, img_array, message=""):
        img = Image.fromarray(img_array) if isinstance(img_array, np.ndarray) else img_array

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 900, 600

        ratio = min(canvas_w / img.width, canvas_h / img.height)
        new_w = int(img.width * ratio * 0.95)
        new_h = int(img.height * ratio * 0.95)

        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=self.photo)

        if message:
            self.info.config(text=message, fg="#27ae60", font=("Arial", 14, "bold"))

    def stop_video(self):
        self.is_playing = False

    def __del__(self):
        self.stop_video()

# ====================== CHẠY APP ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_video(), root.destroy()))
    root.mainloop()
