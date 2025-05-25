# -*- coding: gbk -*-

import sys
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def pixelate_image(input_path, output_path, pixel_size=10, palette_size=16):
    try:
        # 打开图片并转换为 RGB（可处理 jpg/png/去背景的 png）
        with Image.open(input_path) as img:
            img = img.convert('RGBA') if img.mode == 'RGBA' else img.convert('RGB')
        
        # 计算缩略图尺寸
        width, height = img.size
        small_width = max(1, width // pixel_size)
        small_height = max(1, height // pixel_size)

        # 缩小尺寸
        small_img = img.resize((small_width, small_height), Image.NEAREST)

        # 使用 KMeans 进行调色板限定（可选）
        if palette_size > 0:
            pixels = np.array(small_img).reshape(-1, small_img.mode.__len__())
            kmeans = KMeans(n_clusters=palette_size, n_init=1)
            labels = kmeans.fit_predict(pixels)
            quantized = kmeans.cluster_centers_[labels].reshape(small_height, small_width, -1)
            small_img = Image.fromarray(np.clip(quantized, 0, 255).astype('uint8'), mode=small_img.mode)

        # 再次放大到原始尺寸
        result_img = small_img.resize((width, height), Image.NEAREST)

        # 输出 PNG 文件
        result_img.save(output_path, 'PNG')
        print(f"Pixelation is complete! Output file：{output_path}")

    except Exception as e:
        print(f"Fail: {e}")

if __name__ == "__main__":
    # 示例参数（可根据需要更改）
    input_file = "testinput.jpg"
    output_file = "outputpixel.png"
    pixel_size = 3
    palette_size = 32

    pixelate_image(input_file, output_file, pixel_size, palette_size)
