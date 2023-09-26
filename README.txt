# 深度学习鲁棒水印算法，训练后可用于为图像添加水印、检测水印

# 测试
        python -u test.py
    ## 注意
        test.py使用real_tests/shoot_pics中真实拍摄的水印图像进行水印提取准确率的测试，保存至test.csv
        此外，test.py会自动计算嵌入水印的图像与原始图像间的PSNR和SSIM，两类图像分别存储于real_tests/orign_pics和real_tests/encoded_pics
