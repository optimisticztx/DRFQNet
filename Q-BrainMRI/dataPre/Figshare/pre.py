from PIL import Image
import h5py
import numpy as np
import os
# 加载 .mat 文件
for i in range(1,3065):

    mat_data = f"/home/ztx/code/quantumML-code/data/Figshare/data/{i}.mat"

    with h5py.File(mat_data, 'r') as mat_file:
        # 访问数据
        label = mat_file['cjdata']['label'][()]  # 读取标注
        image_data = np.array(mat_file['cjdata']['image'])  # 将数据转换为 numpy 数组
        # pid = mat_file['cjdata']['PID'][()]  # 读取患者ID
        # image_data = mat_file['cjdata']['image'][()]  # 读取图像数据
        # tumor_border = mat_file['cjdata']['tumorBorder'][()]  # 读取肿瘤边界
        # tumor_mask = mat_file['cjdata']['tumorMask'][()]  # 读取肿瘤掩码
        # 创建目录以保存图像
        label_dir = f"./{int(label[0,0])}"  # 输出分类目录
        os.makedirs(label_dir, exist_ok=True)
     # 确保图像数据是 8 位格式 (0-255)
        if image_data.max() > 255:
            image_data = (image_data / image_data.max()) * 255  # 归一化到 0-255 范围
        image_data = image_data.astype(np.uint8)  # 转换为 8 位无符号整型
        # 保存为 .jpg 文件
        image = Image.fromarray(image_data)
        image.save(os.path.join(label_dir, f'{i}.jpg'))  # 替换为你希望保存的文件名
    label_list = {1:"脑膜瘤",2:"胶质瘤",3:"垂体瘤"}
    print("图片：",image_data)
    print("图片size：",image_data.shape)
    print("label：",label[0,0])
    print("类型：",label_list.get(int(label[0,0])))
