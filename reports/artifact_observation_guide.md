# MBRS 实际水印伪影观察图

本次仅推理已有 controlled seed17 checkpoint，使用原 fixed formal test manifest 的50组图像和消息。训练、loss 和既有统计报告未改动。

## 直接打开

- [主对比 PDF：3页，可用于汇报](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/artifact_observation_main.pdf)
- [独立 HTML：全部50张测试图，支持原图切换、1/2/4倍显示](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/artifact_observation.html)
- [样本07：雪地与暗部](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_07.png)
- [样本20：沙地与阴影边缘](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_20.png)
- [样本42：天空与建筑边缘](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/actual_rgb_test_42.png)
- [辅助比较页：加入Top25与Gradient-aware alpha2](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/artifact_observation_supplement.png)

HTML 包含全部图像，无需服务器、网络或额外资源。可下载单文件后用浏览器打开；IDE不支持脚本预览时用PNG/PDF。

## 比较方式

主比较每页三列：原图、仅整图MSE约束的20轮续训模型、Patch16/stride8/Top10且整图/局部损失权重0.5/0.5的20轮续训模型。

每页三行：完整128×128视图、低纹理32×32区域、高纹理32×32区域。完整视图标注取样位置，局部放大显示同一坐标的实际RGB图像。32×32仅为观察窗口，不是改变训练patch或评测协议。

固定展示索引沿用 `[7,20,42]`。局部区域仅由原图决定：在16个不重叠32×32块中，分别选择亮度方差最小与最大的块；相同时取行优先顺序第一块。这种“低/高纹理”是方差代理标签，不表示人眼伪影最轻/最重，也没有按方法收益筛选。

先用HTML的1倍显示查看正常观看尺度，再放大观察颗粒、色彩和边缘；“全部显示原图”按钮让相同位置在原图/模型输出之间切换。辅助方法默认隐藏，主比较始终以原图为共同参照。

## 显示定义与来源

实际张量统一用 `clip((x+1)/2,0,1)` 转为显示RGB，再四舍五入到8-bit；使用无损PNG，放大采用最近邻。没有锐化、局部对比度增强、残差叠加或亮度增益。最近邻块状外观也存在于原图中，不应把放大网格本身解释成模型伪影。

模型输出存在少量越界通道值；固定测试集分别约为：整图MSE续训0.774%、Hard Top10 0.764%、Hard Top25 0.761%、Gradient alpha2 0.784%。全部统一裁剪到合法显示范围。因此这些是可显示图像，不等价于未裁剪浮点张量的误差指标。

manifest SHA-256：`36790b02ca4754f4209539b91b2fe014833001c4202087130ae9c440fbfd5339`。

四个checkpoint均为continuation epoch20。完整checkpoint路径、SHA-256、所有50张图的局部坐标及越界比例见 [provenance.json](/mnt/wmcontent/GLX/icassp/MBRS/visualizations/artifact_observation/provenance.json)。原始128×128 PNG保存在同目录 `native_rgb/`。历史manifest未保存源文件名，展示索引是manifest内部索引，不推测原始DIV2K文件名。

## 直接观察结论

已查看三个固定样本和辅助页。原图与水印图在放大的雪地、天空及暗部可见细小彩色颗粒和色调变化；两种主方法的差异比“原图—水印图”的差异小得多。在这些静态样本上，不能稳定、明确地用肉眼判定Hard Top10全面优于仅整图MSE续训。

这组图适合展示实际输出以及验证伪影是否可见，尚不构成人类主观评分实验。辅助页也仅展示指定区域，不以单张图替代总体评测。

## 复现与验证

运行 `CUDA_VISIBLE_DEVICES=0 /root/miniforge/bin/python experiments/render_artifact_observation.py`。

生成脚本只调用已有编码器推理；主PNG、PDF及辅助PNG已做视觉检查。原尺寸PNG与生成时的显示像素数组做逐像素一致性检查。HTML已检查脚本语法、50张样本数据、前后翻页、原图切换和辅助方法开关（mock DOM）；当前环境未进行真实浏览器截图验证。
