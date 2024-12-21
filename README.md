# Draw Music By Fourier

抽象的音乐，简洁的傅里叶级数，充满想象力的绘画  

使用傅里叶级数绘制音乐

## ManimCE
本项目所提供的demo文件（draw_fourier.py）已适配社区版manim [ManimCE](https://github.com/manimCommunity/manim)   

本项目所需要用到的manim在不同环境下的相关安装与使用方法可在社区中进行查看  


### Installation Manim By Conda（Win/Linux）
以下提供Win/Linux环境下一种简易的安装方法：  

1、安装anaconda，打开anaconda的cmd提示窗口anaconda prompt  

2、输入以下指令：
```python
conda create -n my-manim-environment
conda activate my-manim-environment
conda install -c conda-forge manim
```
其中“my-manim-environment”可更换为自己想要的环境名称，如“mymanim”
### 在manim中运行draw_fourier.py
进入所创建的manim环境，输入指令：
```python
manim -pql draw_fourier.py CustomAnimationExample
```
draw_fourier.py中CustomAnimationExample类所读取的mp3文件可以更换为想要绘画的音频  
（mp3音频文件与draw_fourier.py置于同一目录）

## 参考链接：
[Draw Music By Fourier 用傅里叶绘制音乐_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1HEkaYuEjt/?share_source=copy_web&vd_source=e96f094f5cbc4ce0b1074912888dc399)
