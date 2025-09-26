from ToolBox import (
    read_raw_grayscale, read_bmp_grayscale,
    save_raw_grayscale, save_bmp_grayscale,
    extract_center_block,
    log_transform, gamma_transform, negative_transform,
    bilinear_interpolation, nearest_neighbor_interpolation
)

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class ImageUI:
    def __init__(self, title="HW1 影像處理 UI"):
        # --- 狀態 ---
        self.current_image = None          # 目前影像（在上一張結果上繼續變換）
        self.op_history = []               # 影像快照清單 [img0, img1, ...]，最多 5 張
        self.photo = None                  # Tkinter 需保留參考避免 GC
        self.preview_path = "./__preview.bmp"  # 固定預覽檔

        # --- GUI ---
        self.root = tk.Tk()
        self.root.title(title)

        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(side="left", padx=10, pady=10)

        self.frame = tk.Frame(self.root)
        self.frame.pack(side="right", fill="y", padx=10, pady=10)

        # 參數變數（供 Entry/Scale 綁定）
        self.var_gamma = tk.DoubleVar(value=1.0)
        self.var_bl_w = tk.IntVar(value=256)
        self.var_bl_h = tk.IntVar(value=256)
        self.var_nn_w = tk.IntVar(value=256)
        self.var_nn_h = tk.IntVar(value=256)
        self.var_crop_h = tk.IntVar(value=128)
        self.var_crop_w = tk.StringVar(value="")  # 空字串代表「與高相同」做正方形

        # RAW 載入尺寸（不再跳窗）
        self.var_raw_w = tk.IntVar(value=512)
        self.var_raw_h = tk.IntVar(value=512)

        self._build_controls()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    # ---------- UI ----------
    def _build_controls(self):
        # 區塊：檔案
        section = tk.LabelFrame(self.frame, text="檔案", padx=8, pady=8)
        section.pack(fill="x", pady=6)

        tk.Button(section, text="載入 BMP/RAW", command=self.load_file).pack(fill="x", pady=4)

        raw_row = tk.Frame(section)
        raw_row.pack(fill="x")
        tk.Label(raw_row, text="RAW W").grid(row=0, column=0, sticky="w")
        tk.Entry(raw_row, width=6, textvariable=self.var_raw_w).grid(row=0, column=1, padx=4)
        tk.Label(raw_row, text="RAW H").grid(row=0, column=2, sticky="w")
        tk.Entry(raw_row, width=6, textvariable=self.var_raw_h).grid(row=0, column=3, padx=4)

        # 區塊：基本轉換（無參數）
        section = tk.LabelFrame(self.frame, text="基本轉換", padx=8, pady=8)
        section.pack(fill="x", pady=6)
        tk.Button(section, text="Log Transform", command=lambda: self.apply_op(log_transform)).pack(fill="x", pady=4)
        tk.Button(section, text="Negative", command=lambda: self.apply_op(negative_transform)).pack(fill="x", pady=4)

        # 區塊：Gamma
        section = tk.LabelFrame(self.frame, text="Gamma Transform", padx=8, pady=8)
        section.pack(fill="x", pady=6)

        gamma_row = tk.Frame(section); gamma_row.pack(fill="x")
        tk.Label(gamma_row, text="γ").grid(row=0, column=0, sticky="w")
        tk.Entry(gamma_row, width=8, textvariable=self.var_gamma).grid(row=0, column=1, padx=4)
        # 也提供滑桿方便快速試
        tk.Scale(section, from_=0.1, to=5.0, resolution=0.1,
                 orient="horizontal", variable=self.var_gamma).pack(fill="x", pady=4)
        tk.Button(section, text="套用 Gamma", command=self._apply_gamma).pack(fill="x", pady=4)

        # 區塊：Bilinear Resize
        section = tk.LabelFrame(self.frame, text="Bilinear Resize", padx=8, pady=8)
        section.pack(fill="x", pady=6)

        bl_row = tk.Frame(section); bl_row.pack(fill="x")
        tk.Label(bl_row, text="H").grid(row=0, column=0, sticky="w")
        tk.Entry(bl_row, width=6, textvariable=self.var_bl_h).grid(row=0, column=1, padx=4)
        tk.Label(bl_row, text="w").grid(row=0, column=2, sticky="w")
        tk.Entry(bl_row, width=6, textvariable=self.var_bl_w).grid(row=0, column=3, padx=4)
        tk.Button(section, text="套用 Bilinear", command=self._apply_bilinear).pack(fill="x", pady=4)

        # 區塊：Nearest Resize
        section = tk.LabelFrame(self.frame, text="Nearest Resize", padx=8, pady=8)
        section.pack(fill="x", pady=6)

        nn_row = tk.Frame(section); nn_row.pack(fill="x")
        tk.Label(nn_row, text="H").grid(row=0, column=0, sticky="w")
        tk.Entry(nn_row, width=6, textvariable=self.var_nn_h).grid(row=0, column=1, padx=4)
        tk.Label(nn_row, text="W").grid(row=0, column=2, sticky="w")
        tk.Entry(nn_row, width=6, textvariable=self.var_nn_w).grid(row=0, column=3, padx=4)
        tk.Button(section, text="套用 Nearest", command=self._apply_nearest).pack(fill="x", pady=4)

        # 區塊：Center Crop
        section = tk.LabelFrame(self.frame, text="Center Crop", padx=8, pady=8)
        section.pack(fill="x", pady=6)

        crop_row = tk.Frame(section); crop_row.pack(fill="x")
        tk.Label(crop_row, text="H").grid(row=0, column=0, sticky="w")
        tk.Entry(crop_row, width=6, textvariable=self.var_crop_h).grid(row=0, column=1, padx=4)
        tk.Label(crop_row, text="W（可空）").grid(row=0, column=2, sticky="w")
        tk.Entry(crop_row, width=6, textvariable=self.var_crop_w).grid(row=0, column=3, padx=4)
        tk.Button(section, text="套用 Center Crop", command=self._apply_center_crop).pack(fill="x", pady=4)

        # 區塊：歷史/還原
        section = tk.LabelFrame(self.frame, text="歷史", padx=8, pady=8)
        section.pack(fill="x", pady=6)
        tk.Button(section, text="Undo", command=self.undo).pack(fill="x", pady=4)

    # ---------- 載入 ----------
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="選擇影像",
            filetypes=[("Image files", "*.bmp *.raw"), ("All files", "*.*")]
        )
        if not file_path:
            return

        ext = file_path.lower().split(".")[-1] if "." in file_path else ""
        try:
            if ext == "bmp":
                img = read_bmp_grayscale(file_path)
            elif ext == "raw":
                w = self._get_positive_int(self.var_raw_w.get(), name="RAW 寬度")
                h = self._get_positive_int(self.var_raw_h.get(), name="RAW 高度")
                if w is None or h is None:
                    return
                img = read_raw_grayscale(file_path, w, h)
            else:
                messagebox.showerror("錯誤", "只支援 BMP/RAW")
                return

            # 重置狀態
            self.op_history = [img]
            self.current_image = img
            self._save_preview_and_show(self.current_image)

        except Exception as e:
            messagebox.showerror("錯誤", str(e))

    # ---------- 顯示 ----------
    def _show_image_from_path(self, img_path: str):
        try:
            with Image.open(img_path) as im:
                img = im.copy()  # 關檔避免覆寫時鎖檔（Windows）
        except Exception as e:
            messagebox.showerror("錯誤", f"顯示失敗：{e}")
            return
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def _save_preview_and_show(self, img_array):
        try:
            save_bmp_grayscale(img_array, self.preview_path)
        except Exception as e:
            messagebox.showerror("錯誤", f"暫存失敗：{e}")
            return
        self._show_image_from_path(self.preview_path)

    # ---------- 共用套用器 ----------
    def apply_op(self, func, *args):
        if self.current_image is None:
            messagebox.showinfo("提示", "請先載入影像")
            return
        try:
            new_img = func(self.current_image, *args)
        except Exception as e:
            messagebox.showerror("錯誤", f"運算失敗：{e}")
            return

        self.current_image = new_img
        self.op_history.append(new_img)
        if len(self.op_history) > 5:
            self.op_history = self.op_history[-5:]
        self._save_preview_and_show(self.current_image)

    # ---------- 各操作的「讀參數 + 套用」 ----------
    def _apply_gamma(self):
        gamma = self._get_positive_float(self.var_gamma.get(), name="gamma")
        if gamma is None:
            return
        self.apply_op(gamma_transform, gamma)

    def _apply_bilinear(self):
        w = self._get_positive_int(self.var_bl_w.get(), name="Bilinear 寬度")
        h = self._get_positive_int(self.var_bl_h.get(), name="Bilinear 高度")
        if w is None or h is None:
            return
        self.apply_op(bilinear_interpolation, (h, w))

    def _apply_nearest(self):
        w = self._get_positive_int(self.var_nn_w.get(), name="Nearest 寬度")
        h = self._get_positive_int(self.var_nn_h.get(), name="Nearest 高度")
        if w is None or h is None:
            return
        self.apply_op(nearest_neighbor_interpolation, (h, w))

    def _apply_center_crop(self):
        bh = self._get_positive_int(self.var_crop_h.get(), name="裁切高度")
        if bh is None:
            return
        w_txt = self.var_crop_w.get().strip()
        if w_txt == "":
            bw = bh  # 空 -> 正方形
        else:
            bw = self._get_positive_int(w_txt, name="裁切寬度")
            if bw is None:
                return
        # 假設 ToolBox.extract_center_block(img, (bh, bw))
        self.apply_op(extract_center_block, (bh, bw))

    # ---------- Undo ----------
    def undo(self):
        if len(self.op_history) <= 1:
            messagebox.showinfo("提示", "已在最早狀態，無法回復")
            return
        self.op_history.pop()
        self.current_image = self.op_history[-1]
        self._save_preview_and_show(self.current_image)

    # ---------- 效用：輸入檢查 ----------
    def _get_positive_int(self, v, name="值"):
        try:
            iv = int(v)
            if iv <= 0:
                raise ValueError
            return iv
        except Exception:
            messagebox.showerror("錯誤", f"{name} 必須是正整數")
            return None

    def _get_positive_float(self, v, name="值"):
        try:
            fv = float(v)
            if fv <= 0:
                raise ValueError
            return fv
        except Exception:
            messagebox.showerror("錯誤", f"{name} 必須是正數")
            return None

    # ---------- MainLoop ----------
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ImageUI()
    app.run()
