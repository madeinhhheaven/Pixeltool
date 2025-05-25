# -*- coding: gbk -*-

import sys
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def pixelate_image(input_path, output_path, pixel_size=10, palette_size=16):
    try:
        # ��ͼƬ��ת��Ϊ RGB���ɴ��� jpg/png/ȥ������ png��
        with Image.open(input_path) as img:
            img = img.convert('RGBA') if img.mode == 'RGBA' else img.convert('RGB')
        
        # ��������ͼ�ߴ�
        width, height = img.size
        small_width = max(1, width // pixel_size)
        small_height = max(1, height // pixel_size)

        # ��С�ߴ�
        small_img = img.resize((small_width, small_height), Image.NEAREST)

        # ʹ�� KMeans ���е�ɫ���޶�����ѡ��
        if palette_size > 0:
            pixels = np.array(small_img).reshape(-1, small_img.mode.__len__())
            kmeans = KMeans(n_clusters=palette_size, n_init=1)
            labels = kmeans.fit_predict(pixels)
            quantized = kmeans.cluster_centers_[labels].reshape(small_height, small_width, -1)
            small_img = Image.fromarray(np.clip(quantized, 0, 255).astype('uint8'), mode=small_img.mode)

        # �ٴηŴ�ԭʼ�ߴ�
        result_img = small_img.resize((width, height), Image.NEAREST)

        # ��� PNG �ļ�
        result_img.save(output_path, 'PNG')
        print(f"Pixelation is complete! Output file��{output_path}")

    except Exception as e:
        print(f"Fail: {e}")

if __name__ == "__main__":
    # ʾ���������ɸ�����Ҫ���ģ�
    input_file = "testinput.jpg"
    output_file = "outputpixel.png"
    pixel_size = 3
    palette_size = 32

    pixelate_image(input_file, output_file, pixel_size, palette_size)
