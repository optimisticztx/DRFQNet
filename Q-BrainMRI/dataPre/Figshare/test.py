from PIL import Image
# 打开图像文件
image_path = '/home/ztx/code/quantumML-code/data/BrainTumorMRI/Testing/glioma/Te-gl_0014.jpg'  # 替换为你的图片路径
image = Image.open(image_path)

# 获取图像的大小和通道数
width, height = image.size  # 获取宽度和高度
channels = len(image.getbands())  # 获取通道数

print(f"图像大小: {width}x{height}")
print(f"通道数: {channels}")