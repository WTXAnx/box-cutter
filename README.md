# box-cutter

### 使用说明

1. 将box截图放入img文件夹，支持多张图，需保证多张图使用同一设备截取以保持分辨率一致
2. 使用windows自带的画图，或其它可以用原本分辨率打开图片的软件，打开其中一张box图，使用截图工具截取：
   * 一张卡面；
   * 该卡面上的可以标识卡面的通用指示物，例如无期迷途中可以选取卡图左上角的MBCC，明日方舟中可以选取卡图左下角向内折起的图样；
   * 该卡面上角色名所在的区域，以可能会出现的最长卡名为标准来选取区域，在不截到其他文字的前提下尽量截大一些；
3. 将这三张截图放入template文件夹，直接替换文件夹中的三张示例图片，保持文件名及后缀名一致，jpg文件可以直接重命名修改后缀为png
4. 点击main.exe运行，拆分后的卡图存放在result文件夹
5. 可能出现卡名ocr识别错误或识别不出的情况，建议手动校对生成的文件名
6. 若出现遗漏卡面现象可以适当调低config.json中的置信度阈值，默认为0.95，可以每次减0.05后尝试重新运行
