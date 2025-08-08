import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import torch
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import numpy as np
import json
import cv2
import glob
from pathlib import Path
import sys
from tqdm import tqdm
import lietorch
import torch.nn.functional as F
import argparse
from torch.multiprocessing import Process

sys.path.append('droid_slam')
from droid import Droid
from droid_async import DroidAsync

# Move the Args class here, outside of DroidSLAMIntegrated
class Args:
    def __init__(self, args_dict):
        # Varsayılan değerler
        self.stereo = False
        self.weights = "droid.pth"
        self.buffer = args_dict.get('buffer', 512)
        self.image_size = [240, 320]
        self.disable_vis = args_dict.get('disable_vis', False)
        
        self.beta = args_dict.get('beta', 0.3)
        self.filter_thresh = args_dict.get('filter_thresh', 2.4)
        self.warmup = args_dict.get('warmup', 8)
        self.keyframe_thresh = args_dict.get('keyframe_thresh', 4.0)
        self.frontend_thresh = args_dict.get('frontend_thresh', 16.0)
        self.frontend_window = args_dict.get('frontend_window', 25)
        self.frontend_radius = args_dict.get('frontend_radius', 2)
        self.frontend_nms = args_dict.get('frontend_nms', 1)
        
        self.backend_thresh = args_dict.get('backend_thresh', 22.0)
        self.backend_radius = args_dict.get('backend_radius', 2)
        self.backend_nms = args_dict.get('backend_nms', 3)
        self.upsample = args_dict.get('upsample', True)
        self.asynchronous = args_dict.get('asynchronous', True)
        self.frontend_device = args_dict.get('frontend_device', "cuda")
        self.backend_device = args_dict.get('backend_device', "cuda")
        
        self.reconstruction_path = args_dict.get('reconstruction_path')

class CameraCalibrator:
    """Kamera kalibrasyon yardımcı sınıfı"""
    
    @staticmethod
    def auto_detect_calibration(image_folder):
        """Resim klasöründen otomatik kalibrasyon parametrelerini tahmin et"""
        # İlk resmi yükle
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        
        if not image_files:
            return None
        
        # İlk resmi oku
        img = cv2.imread(image_files[0])
        h, w = img.shape[:2]
        
        # Tipik smartphone/kamera değerleri
        if w > 3000:  # 4K kamera
            fx = fy = w * 0.8  # Tipik değer
        elif w > 1500:  # HD kamera
            fx = fy = w * 0.7
        else:  # Düşük çözünürlük
            fx = fy = w * 0.6
        
        cx = w / 2.0
        cy = h / 2.0
        
        return {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
            'width': w, 'height': h
        }
    
    @staticmethod
    def estimate_motion_blur(image_folder, max_samples=10):
        """Hareket bulanıklığını tahmin et"""
        image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))[:max_samples]
        
        blur_scores = []
        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
            blur_scores.append(laplacian_var)
        
        avg_blur = np.mean(blur_scores)
        return avg_blur


class TrajectoryFilter:
    """Trajectory filtreleme ve düzeltme sınıfı"""
    
    def __init__(self, max_velocity=5.0, max_acceleration=10.0, smoothing_window=5):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.smoothing_window = smoothing_window
        self.positions = []
        self.timestamps = []
        self.outliers = []
    
    def add_position(self, position, timestamp):
        """Yeni pozisyon ekle ve filtrele"""
        if len(self.positions) == 0:
            self.positions.append(position)
            self.timestamps.append(timestamp)
            return position, False
        
        # Hız kontrolü
        dt = timestamp - self.timestamps[-1]
        if dt > 0:
            velocity = np.linalg.norm(position - self.positions[-1]) / dt
            
            if velocity > self.max_velocity:
                # Outlier tespit edildi
                self.outliers.append(len(self.positions))
                # Önceki pozisyonu kullan veya interpolasyon yap
                filtered_pos = self.interpolate_position(position, timestamp)
                self.positions.append(filtered_pos)
                self.timestamps.append(timestamp)
                return filtered_pos, True
        
        self.positions.append(position)
        self.timestamps.append(timestamp)
        return position, False
    
    def interpolate_position(self, current_pos, timestamp):
        """Pozisyon interpolasyonu"""
        if len(self.positions) < 2:
            return self.positions[-1]
        
        # Son iki pozisyon arasında lineer interpolasyon
        last_pos = self.positions[-1]
        prev_pos = self.positions[-2]
        
        # Makul bir pozisyon hesapla
        direction = last_pos - prev_pos
        max_step = self.max_velocity * (timestamp - self.timestamps[-1])
        
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction) * max_step
        
        return last_pos + direction
    
    def smooth_trajectory(self):
        """Trajectory'yi yumuşat"""
        if len(self.positions) < self.smoothing_window:
            return self.positions
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(self.positions)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(self.positions), i + half_window + 1)
            
            window_positions = self.positions[start_idx:end_idx]
            smoothed_pos = np.mean(window_positions, axis=0)
            smoothed.append(smoothed_pos)
        
        return smoothed


class DroidSLAMIntegrated:
    """DROID-SLAM entegre sınıfı"""
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.droid = None
        self.running = False
        self.paused = False
        self.last_position = None 
    def show_image(self, image):
        """Görüntü göster"""
        if not self.gui.disable_vis.get():
            image_np = image.permute(1, 2, 0).cpu().numpy()
            cv2.imshow('DROID-SLAM Live', image_np / 255.0)
            cv2.waitKey(1)
    
    def image_stream(self, imagedir, calib, stride):
        """Görüntü akışı generatörü"""
        calib = np.loadtxt(calib, delimiter=" ")
        fx, fy, cx, cy = calib[:4]

        K = np.eye(3)
        K[0,0] = fx
        K[0,2] = cx
        K[1,1] = fy
        K[1,2] = cy

        image_list = sorted(os.listdir(imagedir))[::stride]

        for t, imfile in enumerate(image_list):
            if not self.running:
                break
                
            while self.paused and self.running:
                time.sleep(0.1)
                
            if not self.running:
                break
                
            image = cv2.imread(os.path.join(imagedir, imfile))

            if image is None:
                print(f"Uyarı: '{imfile}' dosyası okunamadı veya bozuk. Atlanıyor.")
                continue

            if len(calib) > 4:
                image = cv2.undistort(image.copy(), K, calib[4:])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield t, image[None], intrinsics

    def save_reconstruction(self, save_path):
        """Rekonstrüksiyon kaydet"""
        if not self.droid:
            return False
            
        try:
            if hasattr(self.droid, "video2"):
                video = self.droid.video2
            else:
                video = self.droid.video

            t = video.counter.value
            save_data = {
                "tstamps": video.tstamp[:t].cpu(),
                "images": video.images[:t].cpu(),
                "disps": video.disps_up[:t].cpu(),
                "poses": video.poses[:t].cpu(),
                "intrinsics": video.intrinsics[:t].cpu()
            }

            torch.save(save_data, save_path)
            return True
        except Exception as e:
            print(f"Reconstruction kaydetme hatası: {e}")
            return False
    
    def get_current_trajectory(self):
        """Anlık trajectory verisini al"""
        if not self.droid:
            return None
            
        try:
            if hasattr(self.droid, "video2"):
                video = self.droid.video2
            else:
                video = self.droid.video
            
            t = video.counter.value
            if t > 0:
                poses = video.poses[:t].cpu().numpy()
                return poses
        except Exception as e:
            print(f"Trajectory alma hatası: {e}")
        
        return None
    
    def run_slam(self, imagedir, calib_path, args_dict, stop_event):
            """SLAM çalıştır"""
            self.running = True
            
            try:
                args = Args(args_dict)            
                stride = args_dict.get('stride', 3)
                t0 = args_dict.get('t0', 0)
                
                # GUI güncelleme
                self.gui.root.after(0, lambda: self.gui.status_label.config(text="DROID-SLAM başlatılıyor..."))
                
                # İmaj akışı başlat
                frame_count = 0

                for (t, image, intrinsics) in self.image_stream(imagedir, calib_path, stride):
                    if not self.running:
                        break
                        
                    while self.paused and self.running:
                        time.sleep(0.1)
                        if stop_event.is_set():
                            self.running = False
                            break
                        
                    if not self.running:
                        break
                        
                    if t < t0:
                        continue

                    self.show_image(image[0])

                    if self.droid is None:
                        args.image_size = [image.shape[2], image.shape[3]]
                        # DroidAsync yerine Droid'i kullanıyoruz.
                        self.droid = DroidAsync(args) if args.asynchronous else Droid(args)
                        self.gui.root.after(0, lambda: self.gui.status_label.config(text="DROID-SLAM tracking başladı..."))
                    
                    self.droid.track(t, image, intrinsics=intrinsics)
                    frame_count += 1
                    
                    # ############################
                    # POZİSYON FARKINI HESAPLAMA KISMI
                    # ############################
                    current_position = None
                    if hasattr(self.droid, "video2"):
                        video = self.droid.video2
                    else:
                        video = self.droid.video

                    if video.counter.value > 0:
                        # En son pozisyonu al
                        pose = video.poses[video.counter.value-1].cpu().numpy()
                        if pose.ndim == 2 and pose.shape == (4,4):
                            current_position = pose[:3, 3]
                        elif pose.ndim == 1 and pose.shape[0] == 7:
                            current_position = pose[:3]
                    
                    # Sadece pozisyon verisi mevcutsa 0,0,0 noktasına olan uzaklığı hesapla ve yazdır
                    if current_position is not None:
                        print(f"Frame {t}: Konum = (x: {current_position[0]:.4f}, y: {current_position[1]:.4f}, z: {current_position[2]:.4f})")
               
                    
                    # ############################
                    # EKLEDİĞİMİZ KISIM BURADA BİTİYOR
                    # ############################

                    if frame_count % 5 == 0:
                        self.update_gui_with_current_data(t)
                    
                    self.gui.root.after(0, lambda fc=frame_count: 
                                      self.gui.status_label.config(text=f"Processing frame {fc}..."))
                
                if self.running:
                    self.gui.root.after(0, lambda: self.gui.status_label.config(text="SLAM sonlandırılıyor..."))
                    
                    traj_est = self.droid.terminate(self.image_stream(imagedir, calib_path, stride))
                    
                    self.save_final_trajectory(traj_est)
                    
                    if args.reconstruction_path:
                        if self.save_reconstruction(args.reconstruction_path):
                            self.gui.root.after(0, lambda: self.gui.status_label.config(text="✅ SLAM tamamlandı!"))
                        else:
                            self.gui.root.after(0, lambda: self.gui.status_label.config(text="⚠️ SLAM tamamlandı, kaydetme hatası"))
                    else:
                        self.gui.root.after(0, lambda: self.gui.status_label.config(text="✅ SLAM tamamlandı!"))
            
            except Exception as e:
                print(f"SLAM hatası: {e}")
                # Lambda içindeki e değişkenini düzeltmek için
                self.gui.root.after(0, lambda exc=e: self.gui.status_label.config(text=f"❌ SLAM hatası: {exc}"))
            
            finally:
                self.running = False
                cv2.destroyAllWindows()
    
    def update_gui_with_current_data(self, current_frame):
        """Anlık veri ile GUI'yi güncelle"""
        try:
            poses = self.get_current_trajectory()
            if poses is not None and len(poses) > 0:
                # GUI'nin trajectory verilerini güncelle
                self.gui.update_trajectory_from_poses(poses, current_frame)
        except Exception as e:
            print(f"GUI güncelleme hatası: {e}")
    
    def save_final_trajectory(self, traj_est):
        """Final trajectory'yi CSV olarak kaydet"""
        try:
            with open("camera_positions.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Frame", "X", "Y", "Z"])

                origin = None

                for i, T in enumerate(traj_est):
                    if T is None:
                        continue
                    if T.ndim == 1 and T.shape[0] == 7:
                        pos = T[:3]
                    elif T.ndim == 2 and T.shape == (4,4):
                        pos = T[:3, 3]
                    else:
                        print(f"Frame {i}: beklenmedik T shape {T.shape}")
                        continue

                    if origin is None:
                        origin = pos
                    relative_pos = pos - origin
                    writer.writerow([i, relative_pos[0], relative_pos[1], relative_pos[2]])
                    
        except Exception as e:
            print(f"Final trajectory kaydetme hatası: {e}")
    
    def stop(self):
        """SLAM'i durdur"""
        self.running = False
        cv2.destroyAllWindows()
    
    def pause(self):
        """SLAM'i duraklat"""
        self.paused = True
    
    def resume(self):
        """SLAM'i devam ettir"""
        self.paused = False


class DROIDSLAMGUI:
    def __init__(self, root):

        self.root = root
        self.root.title("DROID-SLAM GUI - Entegre Versiyon v3.0")
        self.root.geometry("1400x900")

        # Dosya yolları
        self.video_path = None
        self.image_folder = None
        self.extracted_frames_dir = "temp_frames"
        self.reconstruction_path = "output/reconstruction.pth"
        
        # DROID-SLAM entegrasyonu
        self.droid_slam = DroidSLAMIntegrated(self)
        
        # Trajectory filtreleme
        self.trajectory_filter = TrajectoryFilter()
        self.use_filtering = tk.BooleanVar(value=True)
        
        # Süreç yönetimi
        self.slam_thread = None
        self.stop_event = threading.Event()
        
        # Trajectory verileri
        self.trajectory_data = []
        self.reference_position = None
        self.reference_rotation = None
        
        # Kalibrasyon
        self.auto_calibration = tk.BooleanVar(value=True)
        
        # Grafik verileri
        self.x_data, self.y_data, self.z_data = [], [], []
        self.filtered_x, self.filtered_y, self.filtered_z = [], [], []
        
        # İşleme durumu
        self.expected_frame_count = 0
        self.last_processed_frame = -1
        self.processing_complete = False
        
        # DROID-SLAM ayarları
        self.disable_vis = tk.BooleanVar(value=False)
        
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Ana widget'ları oluştur"""
        # Ana konteyner
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Sol panel - Kontroller
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        
        # Sağ panel - Grafik
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=2)
        
        self.create_control_panel(left_panel)
        self.create_plot_panel(right_panel)

    def create_control_panel(self, parent):
        """Kontrol panelini oluştur"""
        # Dosya seçimi
        file_frame = ttk.LabelFrame(parent, text="Dosya Seçimi")
        file_frame.pack(fill="x", pady=5, padx=5)
        
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(button_frame, text="Video Seç", command=self.select_video).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Resim Klasörü", command=self.select_image_folder).pack(side="left", padx=2)
        
        self.file_label = ttk.Label(file_frame, text="Dosya seçilmedi", foreground="gray")
        self.file_label.pack(padx=10, pady=5)
        
        # Video ayarları
        video_settings_frame = ttk.Frame(file_frame)
        video_settings_frame.pack(fill="x", padx=10, pady=2)
        
        ttk.Label(video_settings_frame, text="Frame Skip:").grid(row=0, column=0, sticky="w")
        self.skip_frames_var = tk.StringVar(value="1")
        ttk.Entry(video_settings_frame, textvariable=self.skip_frames_var, width=6).grid(row=0, column=1, padx=2)
        
        ttk.Label(video_settings_frame, text="Max Frame:").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.max_frames_var = tk.StringVar(value="")
        ttk.Entry(video_settings_frame, textvariable=self.max_frames_var, width=6).grid(row=0, column=3, padx=2)
        
        # Kalibrasyon
        calib_frame = ttk.LabelFrame(parent, text="Kamera Kalibrasyon")
        calib_frame.pack(fill="x", pady=5, padx=5)
        
        ttk.Checkbutton(calib_frame, text="Otomatik Kalibrasyon", 
                       variable=self.auto_calibration, 
                       command=self.toggle_manual_calibration).pack(anchor="w", padx=10, pady=5)
        
        self.manual_calib_frame = ttk.Frame(calib_frame)
        self.manual_calib_frame.pack(fill="x", padx=10, pady=5)
        
        self.calib_entries = {}
        calib_params = [
            ("fx", "1413.3"), ("fy", "950.0639"), ("cx", "1418.8"), ("cy", "543.3796"),
            ("k1", "-0.0091"), ("k2", "0.0666"), ("p1", "0.0"), ("p2", "0.0")
        ]

        for i, (param, default) in enumerate(calib_params):
            row, col = i // 4, (i % 4) * 2
            ttk.Label(self.manual_calib_frame, text=f"{param}:").grid(row=row, column=col, sticky="w", padx=2, pady=1)
            entry = ttk.Entry(self.manual_calib_frame, width=8)
            entry.insert(0, default)
            entry.grid(row=row, column=col+1, padx=2, pady=1)
            self.calib_entries[param] = entry
        
        # DROID-SLAM ayarları
        droid_frame = ttk.LabelFrame(parent, text="DROID-SLAM Ayarları")
        droid_frame.pack(fill="x", pady=5, padx=5)
        
        # Temel ayarlar
        basic_settings = ttk.Frame(droid_frame)
        basic_settings.pack(fill="x", padx=10, pady=5)
        
        # Warmup frames
        ttk.Label(basic_settings, text="Warmup:").grid(row=0, column=0, sticky="w")
        self.warmup_var = tk.StringVar(value="12")
        ttk.Entry(basic_settings, textvariable=self.warmup_var, width=6).grid(row=0, column=1, padx=2)
        
        # Keyframe threshold
        ttk.Label(basic_settings, text="Keyframe Thresh:").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.keyframe_thresh_var = tk.StringVar(value="3.0")
        ttk.Entry(basic_settings, textvariable=self.keyframe_thresh_var, width=6).grid(row=0, column=3, padx=2)
        
        # Filter threshold
        ttk.Label(basic_settings, text="Filter Thresh:").grid(row=1, column=0, sticky="w")
        self.filter_thresh_var = tk.StringVar(value="2.0")
        ttk.Entry(basic_settings, textvariable=self.filter_thresh_var, width=6).grid(row=1, column=1, padx=2)
        
        # Backend threshold
        ttk.Label(basic_settings, text="Backend Thresh:").grid(row=1, column=2, sticky="w", padx=(10,0))
        self.backend_thresh_var = tk.StringVar(value="18.0")
        ttk.Entry(basic_settings, textvariable=self.backend_thresh_var, width=6).grid(row=1, column=3, padx=2)

        async_frame = ttk.Frame(droid_frame)
        async_frame.pack(fill="x", padx=10, pady=5)
        
        self.asynchronous_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(async_frame, text="Asynchronous", variable=self.asynchronous_var).pack(side="left")
        
        ttk.Checkbutton(async_frame, text="Disable Visualization", variable=self.disable_vis).pack(side="left", padx=(20,0))
        
        # Buffer size
        buffer_frame = ttk.Frame(droid_frame)
        buffer_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(buffer_frame, text="Buffer Size:").pack(side="left")
        self.buffer_var = tk.StringVar(value="700")
        ttk.Entry(buffer_frame, textvariable=self.buffer_var, width=8).pack(side="left", padx=5)
        
        ttk.Label(buffer_frame, text="Frontend Window:").pack(side="left", padx=(20,0))
        self.frontend_window_var = tk.StringVar(value="20")
        ttk.Entry(buffer_frame, textvariable=self.frontend_window_var, width=6).pack(side="left", padx=5)
        
        # Filtreleme ayarları
        filter_frame = ttk.LabelFrame(parent, text="Trajectory Filtreleme")
        filter_frame.pack(fill="x", pady=5, padx=5)
        
        ttk.Checkbutton(filter_frame, text="Trajectory Filtreleme Aktif", 
                       variable=self.use_filtering).pack(anchor="w", padx=10, pady=5)
        
        filter_settings = ttk.Frame(filter_frame)
        filter_settings.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(filter_settings, text="Max Hız (m/s):").grid(row=0, column=0, sticky="w")
        self.max_velocity_var = tk.StringVar(value="3.0")
        ttk.Entry(filter_settings, textvariable=self.max_velocity_var, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(filter_settings, text="Smoothing:").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.smoothing_var = tk.StringVar(value="5")
        ttk.Entry(filter_settings, textvariable=self.smoothing_var, width=6).grid(row=0, column=3, padx=2)
        
        # Kontrol butonları
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", pady=10, padx=5)
        
        self.start_button = ttk.Button(button_frame, text="SLAM Başlat", 
                                     command=self.start_slam_thread, 
                                     style="Accent.TButton")
        self.start_button.pack(fill="x", pady=2)
        
        self.stop_button = ttk.Button(button_frame, text="Durdur", 
                                    command=self.stop_slam_process, 
                                    state="disabled")
        self.stop_button.pack(fill="x", pady=2)
        
        self.pause_button = ttk.Button(button_frame, text="Duraklat", 
                                     command=self.pause_slam_process, 
                                     state="disabled")
        self.pause_button.pack(fill="x", pady=2)
        
        ttk.Button(button_frame, text="CSV Kaydet", command=self.export_csv).pack(fill="x", pady=2)
        ttk.Button(button_frame, text="Temizle", command=self.clear_data).pack(fill="x", pady=2)
        
        # Durum paneli
        status_frame = ttk.LabelFrame(parent, text="Durum")
        status_frame.pack(fill="x", pady=5, padx=5)
        
        self.progress = ttk.Progressbar(status_frame, mode='determinate')
        self.progress.pack(fill="x", padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Durum: Hazır", font=("Arial", 9))
        self.status_label.pack(padx=10, pady=2)
        
        self.coord_label = ttk.Label(status_frame, text="Konum: x=0.000, y=0.000, z=0.000", 
                                   font=("Courier", 9), foreground="blue")
        self.coord_label.pack(padx=10, pady=2)

    def create_plot_panel(self, parent):
        """Grafik panelini oluştur"""
        plot_frame = ttk.LabelFrame(parent, text="Trajectory Görselleştirme")
        plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 2x2 subplot düzeni
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # X-Z (Top view)
        self.ax1.set_title("Top View (X-Z)", fontsize=10)
        self.ax1.set_xlabel("X (m)")
        self.ax1.set_ylabel("Z (m)")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_aspect('equal')
        
        # X-Y (Side view)
        self.ax2.set_title("Side View (X-Y)", fontsize=10)
        self.ax2.set_xlabel("X (m)")
        self.ax2.set_ylabel("Y (m)")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_aspect('equal')
        
        # Y-Z (Front view)
        self.ax3.set_title("Front View (Y-Z)", fontsize=10)
        self.ax3.set_xlabel("Y (m)")
        self.ax3.set_ylabel("Z (m)")
        self.ax3.grid(True, alpha=0.3)
        self.ax3.set_aspect('equal')
        
        # 3D view
        self.ax4 = self.fig.add_subplot(224, projection='3d')
        self.ax4.set_title("3D View", fontsize=10)
        self.ax4.set_xlabel("X (m)")
        self.ax4.set_ylabel("Y (m)")
        self.ax4.set_zlabel("Z (m)")
        
        # Çizgiler
        self.line_xz, = self.ax1.plot([], [], 'b-', linewidth=1.5, label='Ham', alpha=0.7)
        self.line_xz_filtered, = self.ax1.plot([], [], 'r-', linewidth=2, label='Filtreli')
        self.point_xz, = self.ax1.plot([], [], 'go', markersize=6, label='Mevcut')
        
        self.line_xy, = self.ax2.plot([], [], 'b-', linewidth=1.5, alpha=0.7)
        self.line_xy_filtered, = self.ax2.plot([], [], 'r-', linewidth=2)
        self.point_xy, = self.ax2.plot([], [], 'go', markersize=6)
        
        self.line_yz, = self.ax3.plot([], [], 'b-', linewidth=1.5, alpha=0.7)
        self.line_yz_filtered, = self.ax3.plot([], [], 'r-', linewidth=2)
        self.point_yz, = self.ax3.plot([], [], 'go', markersize=6)
        
        self.ax1.legend(fontsize=8)
        
        plt.tight_layout()
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def toggle_manual_calibration(self):
        """Manuel kalibrasyon ayarlarını göster/gizle"""
        if self.auto_calibration.get():
            for widget in self.manual_calib_frame.winfo_children():
                widget.configure(state="disabled")
        else:
            for widget in self.manual_calib_frame.winfo_children():
                widget.configure(state="normal")

    def select_video(self):
        """Video dosyası seç"""
        self.video_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        self.image_folder = None
        
        if self.video_path:
            filename = os.path.basename(self.video_path)
            self.file_label.config(text=f"Video: {filename}", foreground="blue")

    def select_image_folder(self):
        """Resim klasörü seç"""
        self.image_folder = filedialog.askdirectory(title="Resim Klasörü Seç")
        self.video_path = None
        
        if self.image_folder:
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))
            
            self.expected_frame_count = len(image_files)
            folder_name = os.path.basename(self.image_folder)
            self.file_label.config(text=f"Klasör: {folder_name} ({len(image_files)} resim)", foreground="blue")
            
            # Otomatik kalibrasyon
            if self.auto_calibration.get():
                self.auto_detect_camera_params()

    def auto_detect_camera_params(self):
        """Otomatik kamera parametrelerini tespit et"""
        if not self.image_folder:
            return
        
        try:
            calib_params = CameraCalibrator.auto_detect_calibration(self.image_folder)
            if calib_params:
                self.calib_entries['fx'].delete(0, tk.END)
                self.calib_entries['fx'].insert(0, f"{calib_params['fx']:.1f}")
                self.calib_entries['fy'].delete(0, tk.END)
                self.calib_entries['fy'].insert(0, f"{calib_params['fy']:.1f}")
                self.calib_entries['cx'].delete(0, tk.END)
                self.calib_entries['cx'].insert(0, f"{calib_params['cx']:.1f}")
                self.calib_entries['cy'].delete(0, tk.END)
                self.calib_entries['cy'].insert(0, f"{calib_params['cy']:.1f}")
                
                self.status_label.config(text=f"Otomatik kalibrasyon: {calib_params['width']}x{calib_params['height']}")
        except Exception as e:
            print(f"Otomatik kalibrasyon hatası: {e}")

    def create_calibration_file(self):
        """Kalibrasyon dosyası oluştur"""
        try:
            calib_params = []
            for key in ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]:
                calib_params.append(float(self.calib_entries[key].get()))
            
            calib_path = "calib/custom_cam.txt"
            os.makedirs("calib", exist_ok=True)
            
            with open(calib_path, "w") as f:
                f.write(" ".join(map(str, calib_params)) + "\n")
            
            return calib_path
            
        except ValueError:
            self.status_label.config(text="Hata: Geçersiz kalibrasyon parametreleri")
            return None
        except IOError as e:
            self.status_label.config(text=f"Kalibrasyon dosyası yazılamadı: {e}")
            return None

    def start_slam_thread(self):
        """SLAM'i thread'de başlat"""
        if self.slam_thread and self.slam_thread.is_alive():
            messagebox.showwarning("Uyarı", "DROID-SLAM zaten çalışıyor!")
            return
        
        self.clear_data()
        self.stop_event.clear()
        
        # UI durumunu güncelle
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.pause_button.config(state="normal")
        
        # Thread başlat
        self.slam_thread = threading.Thread(target=self.run_slam_integrated, daemon=True)
        self.slam_thread.start()

    def run_slam_integrated(self):
        """Entegre SLAM sürecini çalıştır"""
        try:
            input_path = self.video_path if self.video_path else self.image_folder
            if not input_path:
                self.root.after(0, lambda: self.status_label.config(text="Hata: Dosya seçilmedi"))
                return
            
            # Video işleme
            working_dir = input_path
            if self.video_path:
                working_dir = self.process_video_frames()
                if not working_dir:
                    return
            
            # Kalibrasyon dosyası
            calib_path = self.create_calibration_file()
            if not calib_path:
                return
            
            # Output klasörü
            os.makedirs("output", exist_ok=True)
            
            # DROID-SLAM parametreleri hazırla
            args_dict = {
                'buffer': int(self.buffer_var.get()),
                'disable_vis': self.disable_vis.get(),
                'warmup': int(self.warmup_var.get()),
                'keyframe_thresh': float(self.keyframe_thresh_var.get()),
                'filter_thresh': float(self.filter_thresh_var.get()),
                'backend_thresh': float(self.backend_thresh_var.get()),
                'frontend_window': int(self.frontend_window_var.get()),
                'asynchronous': self.asynchronous_var.get(),
                'upsample': True,
                'reconstruction_path': self.reconstruction_path,
                'stride': int(self.skip_frames_var.get()),
                't0': 0
            }
            
            # SLAM'i çalıştır
            self.droid_slam.run_slam(working_dir, calib_path, args_dict, self.stop_event)
            
        except Exception as e:
            self.root.after(0, lambda err=e: self.status_label.config(text=f"Hata: {err}"))
            print(f"SLAM hatası: {e}")
        finally:
            self.root.after(0, self.handle_slam_completion)

    def process_video_frames(self):
        """Video frame işleme"""
        try:
            skip_frames = int(self.skip_frames_var.get()) if self.skip_frames_var.get() else 3
            max_frames = int(self.max_frames_var.get()) if self.max_frames_var.get() else None
            
            # Temp klasörü temizle
            if os.path.exists(self.extracted_frames_dir):
                import shutil
                shutil.rmtree(self.extracted_frames_dir)
            
            frame_count = VideoToFramesConverter.extract_frames(
                self.video_path, self.extracted_frames_dir, max_frames, skip_frames
            )
            
            self.expected_frame_count = frame_count
            self.root.after(0, lambda: self.status_label.config(text=f"{frame_count} frame çıkarıldı"))
            
            return self.extracted_frames_dir
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Frame çıkarma hatası: {e}"))
            return None

    def update_trajectory_from_poses(self, poses, current_frame):
        """Poses verisinden trajectory güncelle"""
        try:
            # İlk frame için referans ayarla
            if self.reference_position is None and len(poses) > 0:
                first_pose = poses[0]
                if len(first_pose.shape) == 2 and first_pose.shape == (4, 4):
                    self.reference_position = first_pose[:3, 3]
                    self.reference_rotation = first_pose[:3, :3]
                elif len(first_pose.shape) == 1 and len(first_pose) >= 3:
                    self.reference_position = first_pose[:3]
                    self.reference_rotation = np.eye(3)
                else:
                    return
            
            # Yeni frame'leri işle
            new_frame_start = max(0, len(self.trajectory_data))
            
            for i in range(new_frame_start, len(poses)):
                pose = poses[i]
                
                # Pozisyon çıkar
                if len(pose.shape) == 2 and pose.shape == (4, 4):
                    current_position = pose[:3, 3]
                elif len(pose.shape) == 1 and len(pose) >= 3:
                    current_position = pose[:3]
                else:
                    continue
                
                # Relatif pozisyon hesapla
                relative_pos = current_position - self.reference_position
                
                # Trajectory filtreleme
                if self.use_filtering.get():
                    max_velocity = float(self.max_velocity_var.get())
                    self.trajectory_filter.max_velocity = max_velocity
                    
                    filtered_pos, is_outlier = self.trajectory_filter.add_position(
                        relative_pos, i
                    )
                    
                    translation_x, translation_y, translation_z = filtered_pos.tolist()
                else:
                    translation_x, translation_y, translation_z = relative_pos.tolist()
                
                frame_name = f"frame_{i:06d}"
                
                frame_data = {
                    'frame': i,
                    'translation_x': translation_x,
                    'translation_y': translation_y,
                    'translation_z': translation_z,
                    'frame_name': frame_name,
                    'is_filtered': self.use_filtering.get()
                }
                
                self.trajectory_data.append(frame_data)
                self.x_data.append(translation_x)
                self.y_data.append(translation_y)
                self.z_data.append(translation_z)
            
            # Smoothing uygula
            if self.use_filtering.get():
                self.apply_smoothing()
            
            # UI güncelle
            self.root.after(0, lambda: self.update_ui_advanced(current_frame))
            
        except Exception as e:
            print(f"Trajectory güncelleme hatası: {e}")

    def apply_smoothing(self):
        """Trajectory smoothing uygula"""
        try:
            if len(self.trajectory_data) < 3:
                return
            
            smoothing_window = int(self.smoothing_var.get())
            
            # Ham veriyi koru
            raw_positions = np.array([[d['translation_x'], d['translation_y'], d['translation_z']] 
                                    for d in self.trajectory_data])
            
            # Smoothing uygula
            smoothed_positions = []
            half_window = smoothing_window // 2
            
            for i in range(len(raw_positions)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(raw_positions), i + half_window + 1)
                
                window_data = raw_positions[start_idx:end_idx]
                smoothed_pos = np.mean(window_data, axis=0)
                smoothed_positions.append(smoothed_pos)
            
            # Filtreli veriyi güncelle
            self.filtered_x = [pos[0] for pos in smoothed_positions]
            self.filtered_y = [pos[1] for pos in smoothed_positions]
            self.filtered_z = [pos[2] for pos in smoothed_positions]
            
        except Exception as e:
            print(f"Smoothing hatası: {e}")

    def update_ui_advanced(self, current_frame):
        """UI güncelleme"""
        # Koordinatları güncelle
        if self.trajectory_data:
            last_data = self.trajectory_data[-1]
            self.coord_label.config(
                text=f"Konum: x={last_data['translation_x']:.3f}, "
                     f"y={last_data['translation_y']:.3f}, "
                     f"z={last_data['translation_z']:.3f}"
            )
        
        # İstatistikleri güncelle
        processed_frames = len(self.trajectory_data)    
        # Progress bar
        if self.expected_frame_count > 0:
            progress_value = min((processed_frames / self.expected_frame_count) * 100, 100)
            self.progress['value'] = progress_value
        
        # Grafikleri güncelle
        self.update_plots_advanced()

    def update_plots_advanced(self):
        """Grafik güncelleme"""
        try:
            if len(self.x_data) == 0:
                return
            
            # Ham veri
            self.line_xz.set_data(self.x_data, self.z_data)
            self.line_xy.set_data(self.x_data, self.y_data)
            self.line_yz.set_data(self.y_data, self.z_data)
            
            # Filtreli veri
            if self.use_filtering.get() and len(self.filtered_x) > 0:
                self.line_xz_filtered.set_data(self.filtered_x, self.filtered_z)
                self.line_xy_filtered.set_data(self.filtered_x, self.filtered_y)
                self.line_yz_filtered.set_data(self.filtered_y, self.filtered_z)
                
                # Mevcut nokta
                if len(self.filtered_x) > 0:
                    self.point_xz.set_data([self.filtered_x[-1]], [self.filtered_z[-1]])
                    self.point_xy.set_data([self.filtered_x[-1]], [self.filtered_y[-1]])
                    self.point_yz.set_data([self.filtered_y[-1]], [self.filtered_z[-1]])
            else:
                # Ham veriden mevcut nokta
                if len(self.x_data) > 0:
                    self.point_xz.set_data([self.x_data[-1]], [self.z_data[-1]])
                    self.point_xy.set_data([self.x_data[-1]], [self.y_data[-1]])
                    self.point_yz.set_data([self.y_data[-1]], [self.z_data[-1]])
            
            # 3D plot güncelle
            self.ax4.clear()
            if self.use_filtering.get() and len(self.filtered_x) > 0:
                self.ax4.plot(self.filtered_x, self.filtered_y, self.filtered_z, 'r-', linewidth=2)
                if len(self.filtered_x) > 0:
                    self.ax4.scatter([self.filtered_x[-1]], [self.filtered_y[-1]], [self.filtered_z[-1]], 
                                   c='g', s=50, marker='o')
            else:
                self.ax4.plot(self.x_data, self.y_data, self.z_data, 'b-', linewidth=1.5)
                if len(self.x_data) > 0:
                    self.ax4.scatter([self.x_data[-1]], [self.y_data[-1]], [self.z_data[-1]], 
                                   c='g', s=50, marker='o')
            
            self.ax4.set_xlabel("X (m)")
            self.ax4.set_ylabel("Y (m)")
            self.ax4.set_zlabel("Z (m)")
            self.ax4.set_title("3D Trajectory")
            
            # Eksen sınırlarını ayarla
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.relim()
                ax.autoscale_view()
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Grafik güncelleme hatası: {e}")

    def handle_slam_completion(self):
        """SLAM tamamlama işlemi"""
        # UI'yi sıfırla
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.pause_button.config(state="disabled")
        self.processing_complete = True
        
        # Otomatik export
        if self.trajectory_data:
            self.export_csv_advanced()

    def export_csv_advanced(self):
        messagebox.showinfo("Başarılı",f"Veriler kaydedildi:\n\n")
        self.status_label.config(text=f"Export tamamlandı:")

    def clear_data(self):
        """Tüm veriyi temizle"""
        self.trajectory_data.clear()
        self.x_data.clear()
        self.y_data.clear()
        self.z_data.clear()
        self.filtered_x = []
        self.filtered_y = []
        self.filtered_z = []
        self.reference_position = None
        self.reference_rotation = None
        self.last_processed_frame = -1
        self.processing_complete = False
        self.trajectory_filter = TrajectoryFilter()
        
        # Grafikleri temizle
        for line in [self.line_xz, self.line_xz_filtered, self.point_xz,
                    self.line_xy, self.line_xy_filtered, self.point_xy,
                    self.line_yz, self.line_yz_filtered, self.point_yz]:
            line.set_data([], [])
        
        self.ax4.clear()
        self.canvas.draw()
        
        # UI sıfırla
        self.coord_label.config(text="Konum: x=0.000, y=0.000, z=0.000")
        self.progress['value'] = 0

    def stop_slam_process(self):
        """SLAM sürecini durdur"""
        self.droid_slam.stop()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.pause_button.config(state="disabled")
        self.status_label.config(text="İşlem durduruldu")

    def pause_slam_process(self):
        """SLAM sürecini duraklat/devam ettir"""
        if self.droid_slam.paused:
            self.droid_slam.resume()
            self.pause_button.config(text="Duraklat")
            self.status_label.config(text="İşlem devam ediyor...")
        else:
            self.droid_slam.pause()
            self.pause_button.config(text="Devam Et")
            self.status_label.config(text="İşlem duraklatıldı")

    def export_csv(self):
        """Eski export fonksiyonu uyumluluğu için"""
        self.export_csv_advanced()

    def cleanup_temp_files(self):
        """Geçici dosyaları temizle"""
        try:
            if os.path.exists(self.extracted_frames_dir):
                import shutil
                shutil.rmtree(self.extracted_frames_dir)
                print("Geçici frame'ler temizlendi")
        except Exception as e:
            print(f"Temp dosya temizleme hatası: {e}")

    def on_closing(self):
        """Uygulama kapatma"""
        if messagebox.askokcancel("Çıkış", 
                                  "Uygulamayı kapatmak istediğinizden emin misiniz?\n"
                                  "Çalışan işlemler durdurulacak."):
            
            if self.slam_thread and self.slam_thread.is_alive():
                print("SLAM işlemi durduruluyor...")
                self.stop_event.set()
                # Thread'in sonlanmasını bekle (max 5 saniye)
                self.slam_thread.join(timeout=5)
            
            self.cleanup_temp_files()
            self.root.destroy()


# VideoToFramesConverter sınıfını da ekleyelim
# VideoToFramesConverter sınıfı
class VideoToFramesConverter:
    """Video'yu frame'lere çeviren yardımcı sınıf"""
    
    @staticmethod
    def extract_frames(video_path, output_dir, max_frames=None, skip_frames=1, target_fps=7.5):
        """Video'dan frame'leri çıkar"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Eğer hedef FPS belirtilmişse, skip_frames değerini otomatik olarak hesapla
        if target_fps and original_fps > target_fps:
            skip_frames = round(original_fps / target_fps)
        elif target_fps:
            skip_frames = 1 # Orijinal FPS daha düşük veya eşitse atlama yapma
        
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    frame_path = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
                
        finally:
            cap.release()
            
        return extracted_count


def main():
    """Ana fonksiyon"""
    root = tk.Tk()
    
    # Tema ayarları
    try:
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Accent.TButton', foreground='white', background='#0066cc')
    except:
        pass
    
    app = DROIDSLAMGUI(root)
    root.mainloop()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()