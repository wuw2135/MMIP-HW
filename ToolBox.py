import os
import math

def bytes_to_int(mbytes, byteorder='little'):
    return int.from_bytes(mbytes, byteorder)

def to_2d_list(data: list[int], height: int, width: int):
    idx = 0
    img = []
    for _ in range(height):
        row = [data[idx + j] for j in range(width)]
        img.append(row)
        idx += width
    return img

#-------------------------------Read-------------------------------
def read_raw_grayscale(path: str, width: int = 512, height: int = 512) -> list[list[int]]:
    with open(path, "rb") as f:
        data = f.read()
    expected_size = width * height
    if len(data) != expected_size:
        raise ValueError(
            f"RAW 大小不符：讀到 {len(data)}，預期 {expected_size} "
            f"對應 {width}x{height} (8-bit)。"
        )
    img = to_2d_list(data, height, width)
    return img

def read_bmp_grayscale(path: str) -> list[list[int]]:
    # from https://blog.csdn.net/qq_36306288/article/details/128692110
    with open(path, "rb") as f:
        bf_type = f.read(2)
        bf_size = f.read(4)
        bf_reserved1 = f.read(2)
        bf_reserved2 = f.read(2)
        bf_off_bits = f.read(4)

        bi_size = f.read(4)
        bi_width = f.read(4)
        bi_height = f.read(4)
        bi_planes = f.read(2)
        bi_bit_count = f.read(2)

        bi_compression = f.read(4)
        bi_size_image = f.read(4)
        bi_x_pels = f.read(4)
        bi_y_pels = f.read(4)
        bi_clr_used = f.read(4)
        bi_clr_important = f.read(4)

        width    = bytes_to_int(bi_width)
        height_s = bytes_to_int(bi_height)      # 有號整數
        bottom_up = height_s > 0
        height   = abs(height_s)

        bitcount    = bytes_to_int(bi_bit_count)
        compression = bytes_to_int(bi_compression)
        if bitcount != 8 or compression != 0:
            raise ValueError(f"僅支援 8-bit、BI_RGB 的 BMP (bitcount={bitcount}, compression={compression})")


        clr_used = bytes_to_int(bi_clr_used) or 256
        pixel_offset = bytes_to_int(bf_off_bits)
        f.seek(pixel_offset, 0)

        row_bytes = width  # 8-bit → 每像素 1 byte
        stride = (row_bytes + 3) // 4 * 4
        pad = stride - row_bytes

        rows = []
        for _ in range(height):
            row_data = f.read(stride)
            if len(row_data) < stride:
                raise ValueError("像素資料不足或檔案損壞")
            # 只取前 width 個 byte 當像素
            rows.append([row_data[x] for x in range(width)])

        # bottom-up → 翻成 top-down，方便後續處理/運算
        if bottom_up:
            rows.reverse()

        return rows

def extract_center_block(img: list[list[int]], block_size: tuple[int, int]) -> list[list[int]]:
    """擷取中心區塊，回傳為 2D list。"""
    h = len(img)
    w = len(img[0])
    bh, bw= block_size
    if h < bh or w < bw: return img
    h_half = bh // 2
    w_half = bw // 2
    cy, cx = h // 2, w // 2
    top = cy - h_half
    left = cx - w_half
    block = []
    for y in range(top, top + bh):
        row = img[y][left:left + bw]
        block.append(row)
    return block


#-------------------------------Save-------------------------------
def validate_image(img: list[list[int]]):
    if not img or not isinstance(img, list) or not isinstance(img[0], list):
        raise ValueError("img 必須是 2D list，例如 list[list[int]]")
    h = len(img)
    w = len(img[0])
    for r, row in enumerate(img):
        if len(row) != w:
            raise ValueError(f"第 {r} 列長度不一致：預期 {w}，實際 {len(row)}")
        for c, v in enumerate(row):
            if not (0 <= int(v) <= 255):
                raise ValueError(f"像素值超出 0..255：({r},{c}) = {v}")
    return h, w

def save_raw_grayscale(img: list[list[int]], path: str):
    h, w = validate_image(img)
    flat = bytearray()
    for row in img:
        flat.extend(int(v) & 0xFF for v in row)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(flat)

def save_bmp_grayscale(img: list[list[int]], path: str):
    h, w = validate_image(img)

    row_bytes = w
    padded_row_bytes = (row_bytes + 3) // 4 * 4
    padding = padded_row_bytes - row_bytes
    image_size = w * h

    file_header_size = 14
    info_header_size = 40
    palette_entries = 256
    palette_size = palette_entries * 4

    bf_off_bits = file_header_size + info_header_size + palette_size
    bf_size = bf_off_bits + image_size

    def u16(x): return int(x).to_bytes(2, "little", signed=False)
    def u32(x): return int(x).to_bytes(4, "little", signed=False)
    def s32(x): return int(x).to_bytes(4, "little", signed=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        f.write(b'BM')
        f.write(u32(bf_size))
        f.write(u16(0))
        f.write(u16(0))
        f.write(u32(bf_off_bits))

        f.write(u32(info_header_size))
        f.write(s32(w))
        f.write(s32(h))
        f.write(u16(1))
        f.write(u16(8))
        f.write(u32(0))
        f.write(u32(image_size))
        f.write(s32(2835))
        f.write(s32(2835))
        f.write(u32(palette_entries))
        f.write(u32(0))

        for i in range(256):
            f.write(bytes((i, i, i, 0)))

        pad_bytes = b"\x00" * padding
        for row in reversed(img):
            f.write(bytes(int(v) & 0xFF for v in row))
            if padding:
                f.write(pad_bytes)

def print_block(block: list[list[int]], name: str):
    print(f"10*10 block in {name}:")
    for row in block:
        print(" ".join(f"{int(v):3d}" for v in row))

#-------------------------------transform-------------------------------
def log_transform(img: list[list[int]]) -> list[list[int]]:
    h, w = len(img), len(img[0])
    out = []
    max_pixel = max(map(max, img))
    scale = 255 / math.log(1 + max_pixel)
    for y in range(h):
        row = []
        for x in range(w):
            s = scale * math.log(1 + img[y][x])
            row.append(int(min(255, max(0, round(s)))))
        out.append(row)
    return out

def negative_transform(img: list[list[int]]) -> list[list[int]]:
    h, w = len(img), len(img[0])
    out = []
    for y in range(h):
        row = [255 - img[y][x] for x in range(w)]
        out.append(row)
    return out

def gamma_transform(img: list[list[int]], gamma: float = 1.0) -> list[list[int]]:
    h, w = len(img), len(img[0])
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            r_norm = img[y][x] / 255.0
            s = (r_norm ** gamma) * 255
            row.append(int(min(255, max(0, round(s)))))
        out.append(row)
    return out

#-------------------------------interpolation-------------------------------
#https://github.com/ChrisCodeNation/Digital-Image-Processing-with-Python/blob/main/4-3%20%E5%BD%B1%E5%83%8F%E6%8F%92%E5%80%BC.ipynb
def bilinear_interpolation(img: list[list[int]], target_hw: tuple[int, int]) -> list[list[int]]:
    h, w = len(img), len(img[0])
    tar_h, tar_w = target_hw
    out = [[0 for _ in range(tar_w)] for _ in range(tar_h)]

    scale_w = float(w) / tar_w
    scale_h = float(h) / tar_h

    for x in range(tar_w):
        for y in range(tar_h):
            # 從原始影像中找離映射點最近的座標值
            org_x = (x+0.5)*scale_w-0.5
            org_y = (y+0.5)*scale_h-0.5
            
            # 計算原圖上四個鄰近點的位置
            org_x0 = int(math.floor(org_x))
            org_y0 = int(math.floor(org_y))
            org_x1 = min(org_x0+1, w-1)
            org_y1 = min(org_y0+1, h-1)
            
            # 雙線性插值
            # 先對兩個x方向進行線性插值
            # 再對y方向進行線性插值
            value0 = (org_x1 - org_x) * img[org_y0][org_x0] + (org_x - org_x0) * img[org_y0][org_x1]
            value1 = (org_x1 - org_x) * img[org_y1][org_x0] + (org_x - org_x0) * img[org_y1][org_x1]
            out[y][x] = int((org_y1 - org_y) * value0 + (org_y - org_y0) * value1)

    return out

def nearest_neighbor_interpolation(img: list[list[int]], target_hw: tuple[int, int]) -> list[list[int]]:
    h, w = len(img), len(img[0])
    tar_h, tar_w = target_hw
    out = [[0 for _ in range(tar_w)] for _ in range(tar_h)]

    for x in range(tar_w):
        for y in range(tar_h):
            sx = x * (w / tar_w)
            sy = y * (h / tar_h)
            org_x = min(max(int(round(sx)), 0), h - 1)
            org_y = min(max(int(round(sy)), 0), w - 1)

            out[y][x] = int(img[org_y][org_x])
    return out

 
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", default="./data")
    parser.add_argument("--crop_folder_path", type=str, default="crop")
    parser.add_argument("--ehz_folder_path", type=str, default="enhance")
    parser.add_argument("--resize_folder_path", type=str, default="resize")
    args = parser.parse_args()

    folder = args.folder_path
    crop_folder = args.crop_folder_path
    enhance_folder = args.ehz_folder_path
    resize_folder = args.resize_folder_path


    files = os.listdir(folder)
        
    result = []
    for f in files:
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):  # 確保是檔案不是資料夾
            name, ext = os.path.splitext(f)
            ext = ext.lower().lstrip('.')  # 去掉前面的「.」
            
            if ext == "raw":
                image = read_raw_grayscale(f"{folder}/{name}.{ext}")
                save_raw_grayscale(image, f"result/{name}/{name}.{ext}")
                img_block = extract_center_block(image, (10,10))
                print_block(img_block, f"{name}.{ext}")
                save_raw_grayscale(img_block, f"result/{name}/{crop_folder}/{name}_crop.{ext}")

                image_log = log_transform(image)
                image_neg = negative_transform(image)
                image_gam_8 = gamma_transform(image, 0.8)
                image_gam_6 = gamma_transform(image, 0.6)
                image_gam_4 = gamma_transform(image, 0.4)
                image_gam_2 = gamma_transform(image, 0.2)
                image_gam_1 = gamma_transform(image, 0.1)
                save_raw_grayscale(image_log, f"result/{name}/{enhance_folder}/{name}_log.{ext}")
                save_raw_grayscale(image_neg, f"result/{name}/{enhance_folder}/{name}_neg.{ext}")
                save_raw_grayscale(image_gam_8, f"result/{name}/{enhance_folder}/{name}_gma_8.{ext}")
                save_raw_grayscale(image_gam_6, f"result/{name}/{enhance_folder}/{name}_gma_6.{ext}")
                save_raw_grayscale(image_gam_4, f"result/{name}/{enhance_folder}/{name}_gma_4.{ext}")
                save_raw_grayscale(image_gam_2, f"result/{name}/{enhance_folder}/{name}_gma_2.{ext}")
                save_raw_grayscale(image_gam_1, f"result/{name}/{enhance_folder}/{name}_gma_1.{ext}")


                image_bi_128 = bilinear_interpolation(image, (128,128))
                image_bi_32 = bilinear_interpolation(image, (32, 32))
                image_bi_512 = bilinear_interpolation(image_bi_32, (512,512))
                image_bi_1024 = bilinear_interpolation(image, (512, 1024))
                image_bi_256 = bilinear_interpolation(image_bi_128, (512, 256))
                save_raw_grayscale(image_bi_128, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_128_128.{ext}")
                save_raw_grayscale(image_bi_32, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_32_32.{ext}")
                save_raw_grayscale(image_bi_512, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_512_512.{ext}")
                save_raw_grayscale(image_bi_1024, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_1024_512.{ext}")
                save_raw_grayscale(image_bi_256, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_256_512.{ext}")


                image_nn_128 = nearest_neighbor_interpolation(image, (128,128))
                image_nn_32 = nearest_neighbor_interpolation(image, (32, 32))
                image_nn_512 = nearest_neighbor_interpolation(image_nn_32, (512,512))
                image_nn_1024 = nearest_neighbor_interpolation(image, (512, 1024))
                image_nn_256 = nearest_neighbor_interpolation(image_nn_128, (512, 256))
                save_raw_grayscale(image_nn_128, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_128_128.{ext}")
                save_raw_grayscale(image_nn_32, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_32_32.{ext}")
                save_raw_grayscale(image_nn_512, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_512_512.{ext}")
                save_raw_grayscale(image_nn_1024, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_1024_512.{ext}")
                save_raw_grayscale(image_nn_256, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_256_512.{ext}")

                #----------------------------------save bmp to preview-------------------------------------
                save_bmp_grayscale(image, f"result/{name}/{name}.bmp")
                save_bmp_grayscale(img_block, f"result/{name}/{crop_folder}/{name}_crop.bmp")
                save_bmp_grayscale(image_log, f"result/{name}/{enhance_folder}/{name}_log.bmp")
                save_bmp_grayscale(image_neg, f"result/{name}/{enhance_folder}/{name}_neg.bmp")
                save_bmp_grayscale(image_gam_8, f"result/{name}/{enhance_folder}/{name}_gma_8.bmp")
                save_bmp_grayscale(image_gam_6, f"result/{name}/{enhance_folder}/{name}_gma_6.bmp")
                save_bmp_grayscale(image_gam_4, f"result/{name}/{enhance_folder}/{name}_gma_4.bmp")
                save_bmp_grayscale(image_gam_2, f"result/{name}/{enhance_folder}/{name}_gma_2.bmp")
                save_bmp_grayscale(image_gam_1, f"result/{name}/{enhance_folder}/{name}_gma_1.bmp")
                save_bmp_grayscale(image_bi_128, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_128_128.bmp")
                save_bmp_grayscale(image_bi_32, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_32_32.bmp")
                save_bmp_grayscale(image_bi_512, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_512_512.bmp")
                save_bmp_grayscale(image_bi_1024, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_1024_512.bmp")
                save_bmp_grayscale(image_bi_256, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_256_512.bmp")
                save_bmp_grayscale(image_nn_128, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_128_128.bmp")
                save_bmp_grayscale(image_nn_32, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_32_32.bmp")
                save_bmp_grayscale(image_nn_512, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_512_512.bmp")
                save_bmp_grayscale(image_nn_1024, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_1024_512.bmp")
                save_bmp_grayscale(image_nn_256, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_256_512.bmp")


            if ext == "bmp":
                image = read_bmp_grayscale(f"{folder}/{name}.{ext}")
                save_bmp_grayscale(image, f"result/{name}/{name}.{ext}")
                img_block = extract_center_block(image, (10,10))
                print_block(img_block, f"{name}.{ext}")
                save_bmp_grayscale(img_block, f"result/{name}/{crop_folder}/{name}_crop.{ext}")

                image_log = log_transform(image)
                image_neg = negative_transform(image)
                image_gam_8 = gamma_transform(image, 0.8)
                image_gam_6 = gamma_transform(image, 0.6)
                image_gam_4 = gamma_transform(image, 0.4)
                image_gam_2 = gamma_transform(image, 0.2)
                image_gam_1 = gamma_transform(image, 0.1)
                save_bmp_grayscale(image_log, f"result/{name}/{enhance_folder}/{name}_log.{ext}")
                save_bmp_grayscale(image_neg, f"result/{name}/{enhance_folder}/{name}_neg.{ext}")
                save_bmp_grayscale(image_gam_8, f"result/{name}/{enhance_folder}/{name}_gma_8.{ext}")
                save_bmp_grayscale(image_gam_6, f"result/{name}/{enhance_folder}/{name}_gma_6.{ext}")
                save_bmp_grayscale(image_gam_4, f"result/{name}/{enhance_folder}/{name}_gma_4.{ext}")
                save_bmp_grayscale(image_gam_2, f"result/{name}/{enhance_folder}/{name}_gma_2.{ext}")
                save_bmp_grayscale(image_gam_1, f"result/{name}/{enhance_folder}/{name}_gma_1.{ext}")

                image_bi_128 = bilinear_interpolation(image, (128,128))
                image_bi_32 = bilinear_interpolation(image, (32, 32))
                image_bi_512 = bilinear_interpolation(image_bi_32, (512,512))
                image_bi_1024 = bilinear_interpolation(image, (512, 1024))
                image_bi_256 = bilinear_interpolation(image_bi_128, (512, 256))
                save_bmp_grayscale(image_bi_128, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_128_128.{ext}")
                save_bmp_grayscale(image_bi_32, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_32_32.{ext}")
                save_bmp_grayscale(image_bi_512, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_512_512.{ext}")
                save_bmp_grayscale(image_bi_1024, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_1024_512.{ext}")
                save_bmp_grayscale(image_bi_256, f"result/{name}/{resize_folder}/bilinear_interpolation/{name}_bi_256_512.{ext}")


                image_nn_128 = nearest_neighbor_interpolation(image, (128,128))
                image_nn_32 = nearest_neighbor_interpolation(image, (32, 32))
                image_nn_512 = nearest_neighbor_interpolation(image_nn_32, (512,512))
                image_nn_1024 = nearest_neighbor_interpolation(image, (512, 1024))
                image_nn_256 = nearest_neighbor_interpolation(image_nn_128, (512, 256))
                save_bmp_grayscale(image_nn_128, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_128_128.{ext}")
                save_bmp_grayscale(image_nn_32, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_32_32.{ext}")
                save_bmp_grayscale(image_nn_512, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_512_512.{ext}")
                save_bmp_grayscale(image_nn_1024, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_1024_512.{ext}")
                save_bmp_grayscale(image_nn_256, f"result/{name}/{resize_folder}/nearest_neighbor_interpolation/{name}_nn_256_512.{ext}")
            



            
