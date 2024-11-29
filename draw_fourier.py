from manim import *
import itertools as it
from _functools import reduce
import operator as op
import os


# import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np

from scipy.io import wavfile
from scipy.fft import fft
from scipy.fft import ifft
from scipy.interpolate import interp1d
from scipy.io.wavfile import write

#################################################
#  __  __             _            _____ ______
# |  \/  |           (_)          / ____|  ____|
# | \  / | __ _ _ __  _ _ __ ___ | |    | |__
# | |\/| |/ _` | '_ \| | '_ ` _ \| |    |  __|
# | |  | | (_| | | | | | | | | | | |____| |____
# |_|  |_|\__,_|_| |_|_|_| |_| |_|\_____|______|
################################################

#    / \  | |__  ___| |_ _ __ __ _  ___| |_
#   / _ \ | '_ \/ __| __| '__/ _` |/ __| __|
#  / ___ \| |_) \__ \ |_| | | (_| | (__| |_
# /_/   \_\_.__/|___/\__|_|  \__,_|\___|\__|
#  ____
# / ___|  ___ ___ _ __   ___  ___
# \___ \ / __/ _ \ '_ \ / _ \/ __|
#  ___) | (_|  __/ | | |  __/\__ \
# |____/ \___\___|_| |_|\___||___/
########################################################################################

# 4个主要的抽象类，第一个是最更根本的
class FourierCirclesScene(ZoomedScene):

   def __init__(
           self,
           n_vectors=10,
           big_radius=2,
           colors=[
               BLUE_D,
               BLUE_C,
               BLUE_E,
               GREY_BROWN,
           ],                      # 圆圈的颜色
           vector_config={
               'buff': 0,
               'max_tip_length_to_length_ratio': 0.25,
               'tip_length': 0.15,
               'max_stroke_width_to_length_ratio': 8,
               'stroke_width': 1.7,
           },
           circle_config={
               'stroke_width': 1
           },
           base_frequency=1,
           slow_factor=0.5,
           center_point=ORIGIN,
           parametric_function_step_size=0.001,
           drawn_path_color=YELLOW,            # 绘制过程中线的颜色
           drawn_path_stroke_width=2,          #
           interpolate_config=[0, 1],
           # Zoom config
           include_zoom_camera=False,
           scale_zoom_camera_to_full_screen=False,
           scale_zoom_camera_to_full_screen_at=4,
           zoom_factor=0.8,                        # 缩放因子：越小放大的区域越小，越小放大倍数越大
           zoomed_display_height=3,
           zoomed_display_width=4,
           image_frame_stroke_width=20,
           zoomed_camera_config={
               "default_frame_stroke_width": 1,    # 放大框的线粗细
               "background_opacity": 1,            # 放大框的不透明度
               'cairo_line_width_multiple': 0.01
           },
           zoom_position=lambda mob: mob.move_to(ORIGIN),
           zoom_camera_to_full_screen_config={
               "run_time": 3,
               "func": there_and_back_with_pause,
               "velocity_factor": 1,
           },
           wait_before_start=None,
           **kwargs,
   ):
       self.n_vectors = n_vectors
       self.big_radius = big_radius
       self.colors = colors #
       self.vector_config = vector_config
       self.circle_config = circle_config
       self.base_frequency = base_frequency #
       self.slow_factor = slow_factor
       self.center_point = center_point
       self.parametric_function_step_size = parametric_function_step_size
       self.drawn_path_color = drawn_path_color
       self.drawn_path_stroke_width = drawn_path_stroke_width
       self.interpolate_config = interpolate_config
       self.include_zoom_camera = include_zoom_camera
       self.scale_zoom_camera_to_full_screen = scale_zoom_camera_to_full_screen
       self.scale_zoom_camera_to_full_screen_at = scale_zoom_camera_to_full_screen_at
       self.zoom_position = zoom_position
       self.zoom_camera_to_full_screen_config = zoom_camera_to_full_screen_config
       self.wait_before_start = wait_before_start

       super().__init__(
           zoom_factor=zoom_factor,
           zoomed_display_height=zoomed_display_height,
           zoomed_display_width=zoomed_display_width,
           image_frame_stroke_width=image_frame_stroke_width,
           zoomed_camera_config=zoomed_camera_config,
           **kwargs
       )

   def setup(self):
       ZoomedScene.setup(self)
       # 设置一个速度变量，初始值设置为slow_factor
       self.slow_factor_tracker = ValueTracker(
           self.slow_factor
       )
       self.vector_clock = ValueTracker(0)  #设置一个时间变量名为vector_clock，并将其初始值设置为0
       # self.add（）将对象vector_clock添加到self中
       self.add(self.vector_clock)

   def add_vector_clock(self):
       self.vector_clock.add_updater(
           lambda m, dt: m.increment_value(
               self.get_slow_factor() * dt
           )
       )

   def get_slow_factor(self):
       return self.slow_factor_tracker.get_value()

   def get_vector_time(self):
       return self.vector_clock.get_value()

   def get_freqs(self):
       n = self.n_vectors
       # 取正负对称的频率点
       all_freqs = list(range(n // 2, -n // 2, -1))
       all_freqs.sort(key=abs)
       return all_freqs

   def get_coefficients(self):
       # 创建n_vectors个复数0（0+0j）
       return [complex(0) for _ in range(self.n_vectors)]

   def get_color_iterator(self):
       return it.cycle(self.colors)

   def get_rotating_vectors(self, freqs=None, coefficients=None):
       vectors = VGroup()
       self.center_tracker = VectorizedPoint(self.center_point)

       if freqs is None:
           freqs = self.get_freqs()
       if coefficients is None:
           coefficients = self.get_coefficients()

       last_vector = None
       for freq, coefficient in zip(freqs, coefficients):
           # center可以理解为上一个向量的末尾坐标，也是下一个向量的起始坐标
           if last_vector:
               center_func = last_vector.get_end
           else:
               center_func = self.center_tracker.get_location
           vector = self.get_rotating_vector(
               coefficient=coefficient,
               freq=freq,
               center_func=center_func,
           )
           # 添加 -> jermain
           #? 为什么要减去get_start()，get_start()有没有可能一直是（0，0）？
           #! 一般情况下get_start()是(0,0)，如果绘画的中点定在(0,0)的话，但如果整个画面进行平移就需要用get_start()保证平移量始终与对应的前一个向量对应
           vector.shift(vector.center_func() - vector.get_start())
           vectors.add(vector)
           last_vector = vector
       return vectors
   

   def get_rotating_vector(self, coefficient, freq, center_func):
       # 建立一个方向向右，长度为coefficient的绝对值的向量vector，向量根据vector_config进行配置
       vector = Vector(RIGHT * abs(coefficient), **self.vector_config)

       if abs(coefficient) == 0:
           phase = 0
       else:
           # 对复数进行自然对数求解，欧拉公式e^(iθ)=cosθ+isinθ，其中coef系数(a+bj)可用等式右边表示
           # 对欧拉公式两边取自然对数：log(e^(iθ))=log(cosθ+isinθ)=iθ
           # 注意这里的e^(iθ)中θ为变量而且可能θ为复数，所以iθ包括了幅值和相位
           # 取iθ=i*（a+bj）的虚部就能得到coef系数在极坐标下的相位(弧度制表示，即向量与水平线的夹角角度θ)，实部为幅度/幅值
           phase = np.log(coefficient).imag            # 取得Fn的相位 Im[ln(Fn)]

       # 向量Fn，根据coef系数的相位旋转对应的角度
       # phase是给定的相位值，用于确定旋转的角度；about_point=ORIGIN 指定了旋转的中心点为原点
       vector.rotate(phase, about_point=ORIGIN)

       vector.freq = freq
       vector.coefficient = coefficient
       vector.center_func = center_func # 上一个向量的末尾坐标.get_end()
       # vector根据更新器函数update_vector对vector内的属性进行更新
       vector.add_updater(self.update_vector)
       return vector

   def update_vector(self, vector, dt):
       time = self.get_vector_time()
       coef = vector.coefficient
       freq = vector.freq
       phase = np.log(coef).imag

       vector.set_length(abs(coef))
       # 根据时间的值去修改相位（相位随着时间t按照对应的频率进行变化，这里用时间t乘以频率f，2Π/w = 2Πf，w为角频率，TAU是2Π的常量符号）
       vector.set_angle(phase + time * freq * TAU)
       # 这句话将向量移动到对应的位置（例如第2个向量的起初坐标是第1个向量的末尾，第7个向量的起始坐标是第6个的末尾）
       vector.shift(vector.center_func() - vector.get_start())
       #注意这里vector进行shift移动后，get_start()的值发生改变，变成移动后的值即center_func()
       return vector

   def get_circles(self, vectors):
       return VGroup(*[
           self.get_circle(
               vector,
               color=color
           )
           for vector, color in zip(
               vectors,
               self.get_color_iterator()
           )
       ])

   def get_circle(self, vector, color=BLUE):
       circle = Circle(color=color, **self.circle_config)
       circle.center_func = vector.get_start
       circle.radius_func = vector.get_length
       # 添加 -> jermain
       # circle.scale_to_fit_width()方法将圆缩放到给定的宽度，即直径的两倍，以确保其宽度与给定值匹配
       circle.scale_to_fit_width(2 * circle.radius_func())
       # 将当前向量vector对应的circle圆圈移动到上一个向量的末尾坐标，起始坐标是当前向量对应圆的圆心
       circle.shift(circle.center_func())

       circle.add_updater(self.update_circle)
       return circle

   def update_circle(self, circle):
       # jermain
       # circle.scale_to_fit_width()方法将圆缩放到给定的宽度，即直径的两倍，以确保其宽度与给定值匹配
       #? 这里为什么要更新圆呢，因为多段路径的话会有多个组合向量，所以圆圈需要进行更新
       circle.scale_to_fit_width(2 * circle.radius_func())
       circle.move_to(circle.center_func())
       return circle

   def get_vector_sum_path(self, vectors, color=YELLOW):
       coefs = [v.coefficient for v in vectors]
       freqs = [v.freq for v in vectors]
       # 绘画的中心center是第一个向量的初始坐标即起点
       center = vectors[0].get_start()

       # reduce(op.add, [...]) 的作用是对 ... 中的元素进行累加，得到一个复数。
       # complex_to_R3函数是一个自定义函数，用于将复数转换为Manim中的三维空间中的点（R3）。这个函数的作用是将复数的实部映射到x轴，虚部映射到y轴，然后返回一个三维向量，其z分量为0。
       # 定义了一个 ParametricFunction，它通过参数 t 生成了一个复数序列，并将其转换为三维空间的向量序列。
       # 每个复数由多个频率和系数的线性组合得到，然后通过 complex_to_R3 函数将其转换为三维空间中的点。最后，这些点被连接起来以生成 ParametricFunction。
       path = ParametricFunction(
           lambda t: center + reduce(op.add, [
               complex_to_R3(
                   coef * np.exp(TAU * 1j * freq * t)
               )
               for coef, freq in zip(coefs, freqs)
           ]),
           t_range=[0, 1, self.parametric_function_step_size],
           color=color,
       )
       return path

   def get_drawn_path_alpha(self):
       return self.get_vector_time()

   def get_drawn_path(self, vectors, stroke_width=None, **kwargs):
       if stroke_width is None:
           stroke_width = self.drawn_path_stroke_width
       path = self.get_vector_sum_path(vectors, **kwargs)

       # CurvesAsSubmobjects是一个函数，用于将路径（Path）对象拆分为多个子对象（Submobjects）
       # CurvesAsSubmobjects接受一个路径对象作为参数，并返回一个由路径中的曲线组成的子对象列表，可以对这些子对象进行单独的控制，例如改变它们的颜色、线宽等
       broken_path = CurvesAsSubmobjects(path)
       broken_path.curr_time = 0
       # start，end为0和1
       start, end = self.interpolate_config

       # 根据时间更新路径path，时间刻度再当前时间之前的路径进行显示，时间刻度在当前时间之后的设置线宽为0从而不显示
       def update_path(path, dt):
           alpha = self.get_drawn_path_alpha()
           n_curves = len(path)
           for a, sp in zip(np.linspace(0, 1, n_curves), path):
               # 用b来判断当前时刻alpha跟所有时间刻度的对比，大于0则是当前时刻之前，小于0是当前时刻之后
               b = (alpha - a)
               if b < 0:
                   width = 0
               else:
                   #? 通过对b去模1控制b在[0，1]内，但对长度实现何种功能有待探究
                   width = stroke_width * interpolate(start, end, (1 - (b % 1)))
               sp.set_stroke(width=width) 
           path.curr_time += dt
           return path

       broken_path.set_color(self.drawn_path_color)
       broken_path.add_updater(update_path)
       return broken_path
   
   def get_drawn_path_py(self, vectors, stroke_width=None, **kwargs):
       if stroke_width is None:
           stroke_width = self.drawn_path_stroke_width
       path = self.get_vector_sum_path(vectors, **kwargs)

       # CurvesAsSubmobjects是一个函数，用于将路径（Path）对象拆分为多个子对象（Submobjects）
       # CurvesAsSubmobjects接受一个路径对象作为参数，并返回一个由路径中的曲线组成的子对象列表，可以对这些子对象进行单独的控制，例如改变它们的颜色、线宽等
       broken_path = CurvesAsSubmobjects(path)
       broken_path.curr_time = 0
       # start，end为0和1
       start, end = self.interpolate_config

       # 根据时间更新路径path，时间刻度再当前时间之前的路径进行显示，时间刻度在当前时间之后的设置线宽为0从而不显示
    #    def update_path(path, dt):
    #        alpha = self.get_drawn_path_alpha()
    #        n_curves = len(path)
    #        for a, sp in zip(np.linspace(0, 1, n_curves), path):
    #            # 用b来判断当前时刻alpha跟所有时间刻度的对比，大于0则是当前时刻之前，小于0是当前时刻之后
    #            b = (alpha - a)
    #            if b < 0:
    #                width = 0
    #            else:
    #                #? 通过对b去模1控制b在[0，1]内，但对长度实现何种功能有待探究
    #                width = stroke_width * interpolate(start, end, (1 - (b % 1)))
    #            sp.set_stroke(width=width) 
    #        path.curr_time += dt
    #        return path

       broken_path.set_color(self.drawn_path_color)
    #    broken_path.add_updater(update_path)
       return broken_path
   

   def get_y_component_wave(self,
                            vectors,
                            left_x=1,
                            color=PINK,
                            n_copies=2,
                            right_shift_rate=5):
       path = self.get_vector_sum_path(vectors)
       wave = ParametricFunction(
           # 将向左LEFT变化的坐标与向上UP变化的坐标相加
           lambda t: op.add(
               right_shift_rate * t * LEFT,
               path.function(t)[1] * UP
           ),
           t_min=path.t_min,
           t_max=path.t_max,
           color=color,
       )
       wave_copies = VGroup(*[
           wave.copy()
           for x in range(n_copies)
       ])
       # 将复制之后的wave列表按照水平向右进行排列
       wave_copies.arrange(RIGHT, buff=0)
       # .get_top()获取对象的顶部边界的y坐标值
       top_point = wave_copies.get_top()
       # 通过调用wave.creation来执行创建wave物体的动画
       wave.creation = Create(
           wave,
           run_time=(1 / self.get_slow_factor()),
           rate_func=linear,
       )
       # cycle_animation()函数用于循环播放动画。它接受一个动画对象作为参数，并将该动画对象设置为循环播放
       cycle_animation(wave.creation)
       # wave图像在每一帧更新的时候逐渐左移
       wave.add_updater(lambda m: m.shift(
           (m.get_left()[0] - left_x) * LEFT
       ))

       def update_wave_copies(wcs):
           index = int(
               wave.creation.total_time * self.get_slow_factor()
           )
           wcs[:index].match_style(wave)
           wcs[index:].set_stroke(width=0)
           wcs.next_to(wave, RIGHT, buff=0)
           wcs.align_to(top_point, UP)
       wave_copies.add_updater(update_wave_copies)

       return VGroup(wave, wave_copies)

   def get_wave_y_line(self, vectors, wave):
       # DashedLine是一个类，用于创建一个虚线
       # DashedLine的参数是两个点的坐标，表示虚线的起点和终点
       return DashedLine(
           vectors[-1].get_end(),
           wave[0].get_end(),
           stroke_width=1,
           dash_length=DEFAULT_DASH_LENGTH * 0.5,
       )

   def get_coefficients_of_path(self, path, n_samples=10000, freqs=None):
       if freqs is None:
           freqs = self.get_freqs()
       dt = 1 / n_samples
       ts = np.arange(0, 1, dt)
       samples = np.array([
           # 用于获取路径对象（Path）上给定比例位置处的点的坐标
           path.point_from_proportion(t)
           for t in ts
       ])
       samples -= self.center_point
       complex_samples = samples[:, 0] + 1j * samples[:, 1]
       return [
           np.array([
               np.exp(-TAU * 1j * freq * t) * cs
               for t, cs in zip(ts, complex_samples)
           ]).sum() * dt for freq in freqs
       ]

   def zoom_config(self):
       # This is not in the original version of the code.
       self.activate_zooming(animate=False)
       self.zoom_position(self.zoomed_display)
       self.zoomed_camera.frame.add_updater(lambda mob: mob.move_to(self.vectors[-1].get_end()))

   def scale_zoom_camera_to_full_screen_config(self):
       # This is not in the original version of the code.
       def fix_update(mob, dt, velocity_factor, dt_calculate):
           if dt == 0 and mob.counter == 0:
               rate = velocity_factor * dt_calculate
               mob.counter += 1
           else:
               rate = dt * velocity_factor
           if dt > 0:
               mob.counter = 0
           return rate

       fps = 1 / self.camera.frame_rate        #
       mob = self.zoomed_display
       mob.counter = 0
       velocity_factor = self.zoom_camera_to_full_screen_config["velocity_factor"]
       mob.start_time = 0
       run_time = self.zoom_camera_to_full_screen_config["run_time"]
       # jermain
       mob_height = mob.get_height()
       mob_width = mob.get_width()
       mob_center = mob.get_center()
       ctx = self.zoomed_camera.cairo_line_width_multiple

       def update_camera(mob, dt):
           line = Line(
               mob_center,
               # self.camera_frame.get_center()
               # 修改 -> jermain
               self.camera.frame_center,
           )
           mob.start_time += fix_update(mob, dt, velocity_factor, fps)
           if mob.start_time <= run_time:
               alpha = mob.start_time / run_time
               alpha_func = self.zoom_camera_to_full_screen_config["func"](alpha)
               coord = line.point_from_proportion(alpha_func)

               mob.set_height(
                   interpolate(
                       mob_height,
                       # self.camera_frame.get_height(),
                       # 修改 -> jermain
                       self.camera.frame_height,
                       alpha_func
                   ),
                   # 当前还没有排查为什么加入会报错
                   # stretch=True
               )
               mob.set_width(
                   interpolate(
                       mob_width,
                       # self.camera_frame.get_width(),
                       # 修改 -> jermain
                       self.camera.frame_width,
                       alpha_func
                   ),
                   # 当前还没有排查为什么加入会报错
                   # stretch=True
               )
               self.zoomed_camera.cairo_line_width_multiple = interpolate(
                   ctx,
                   self.camera.cairo_line_width_multiple,
                   alpha_func
               )
               mob.move_to(coord)
           return mob

       self.zoomed_display.add_updater(update_camera)


class AbstractFourierOfTexSymbol(FourierCirclesScene):
   def __init__(
           self,
           n_vectors=50,
           center_point=ORIGIN,
           slow_factor=0.05,
           n_cycles=None,
           run_time=10,
           tex=r"\rm M",
           start_drawn=True,
           path_custom_position=lambda mob: mob,
           max_circle_stroke_width=1,
           tex_class=Tex,
           tex_config={
               "fill_opacity": 0,
               "stroke_width": 1,
               "stroke_color": WHITE
           },
           include_zoom_camera=False,
           scale_zoom_camera_to_full_screen=False,
           scale_zoom_camera_to_full_screen_at=1,
           zoom_position=lambda mob: mob.scale(0.8).move_to(ORIGIN).to_edge(RIGHT),
           **kwargs,
   ):

       self.n_cycles = n_cycles
       self.run_time = run_time
       self.tex = tex
       self.start_drawn = start_drawn
       self.path_custom_position = path_custom_position
       self.max_circle_stroke_width = max_circle_stroke_width
       self.tex_class = tex_class
       self.tex_config = tex_config
       self.include_zoom_camera = include_zoom_camera
       self.scale_zoom_camera_to_full_screen = scale_zoom_camera_to_full_screen
       self.scale_zoom_camera_to_full_screen_at = scale_zoom_camera_to_full_screen_at
       # self.zoom_position = zoom_position

       super().__init__(
           n_vectors=n_vectors,
           center_point=center_point,
           slow_factor=slow_factor,
           include_zoom_camera=include_zoom_camera,
           zoom_position=zoom_position,
           scale_zoom_camera_to_full_screen=scale_zoom_camera_to_full_screen,
           scale_zoom_camera_to_full_screen_at=scale_zoom_camera_to_full_screen_at,
           **kwargs
       )

   def construct(self):
       # This is not in the original version of the code.
       self.add_vectors_circles_path()
       if self.wait_before_start != None:
           self.wait(self.wait_before_start)
       self.add_vector_clock()
       self.add(self.vector_clock)
       if self.include_zoom_camera:
           self.zoom_config()
       # # jermain
       if self.n_cycles:
           if not self.scale_zoom_camera_to_full_screen:
               for n in range(self.n_cycles):
                   self.run_one_cycle()
           else:
               cycle = 1 / self.slow_factor
               total_time = cycle * self.n_cycles
               total_time -= self.scale_zoom_camera_to_full_screen_at
               self.wait(self.scale_zoom_camera_to_full_screen_at)
               self.scale_zoom_camera_to_full_screen_config()
               self.wait(total_time)

       elif not self.n_cycles and self.run_time:
           if self.scale_zoom_camera_to_full_screen:
               self.run_time -= self.scale_zoom_camera_to_full_screen_at
               self.wait(self.scale_zoom_camera_to_full_screen_at)
               self.scale_zoom_camera_to_full_screen_config()
           self.wait(self.run_time)

   def add_vectors_circles_path(self):
       path = self.get_path()
       self.path_custom_position(path) #
       coefs = self.get_coefficients_of_path(path)

       vectors = self.get_rotating_vectors(coefficients=coefs)
       circles = self.get_circles(vectors)
       self.set_decreasing_stroke_widths(circles)
       drawn_path = self.get_drawn_path(vectors)

       if self.start_drawn:
           # increment_value表示增量值
           self.vector_clock.increment_value(1)

       self.add(path)
       self.add(vectors)
       self.add(circles)
       self.add(drawn_path)

       self.vectors = vectors
       self.circles = circles
       self.path = path
       self.drawn_path = drawn_path

   def run_one_cycle(self):
       time = 1 / self.slow_factor
       self.wait(time)

   def set_decreasing_stroke_widths(self, circles):
       mcsw = self.max_circle_stroke_width
       # it.count(1)是通过使用itertools模块中的count函数创建了一个无限迭代器，该迭代器从1开始以步长1无限地生成整数序列
       for k, circle in zip(it.count(1), circles):
           circle.set_stroke(width=max(
               mcsw / k,
               mcsw,
           ))
       return circles

   def get_path(self):
       tex_mob = self.tex_class(self.tex, **self.tex_config)
       tex_mob.set_height(6)
       # tex_mob.scale_to_fit_height(6)
       # 通过tex_mob.family_members_with_points()方法获取了TeX文本对象中的路径信息
       path = tex_mob.family_members_with_points()[0]
       return path

class AbstractFourierFromSVG(AbstractFourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=101,
           run_time=10,
           start_drawn=True,
           file_name=None,
           svg_config={
               "fill_opacity": 0,
               "stroke_color": WHITE,
               "stroke_width": 1,
               "height": 7
           },
           **kwargs
   ):
       self.file_name = file_name
       self.svg_config = svg_config
       super().__init__(
           n_vectors=n_vectors,
           run_time=10,
           start_drawn=start_drawn,
           file_name=file_name,
           svg_config=svg_config,
           **kwargs
       )

   def get_shape(self):
       shape = SVGMobject(self.file_name, **self.svg_config)
       return shape

   def get_path(self):
       shape = self.get_shape()
       path = shape.family_members_with_points()[0]
       return path


class FourierOfPaths(AbstractFourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=100,
           name_color=WHITE,  # 这是个啥
           tex_class=Tex,
           tex=None,
           file_name=None,
           tex_config={
               "stroke_color": WHITE,
               "fill_opacity": 0,
               "stroke_width": 3,
           },
           svg_config={},
           time_per_symbol=5,
           slow_factor=0.2,
           parametric_function_step_size=0.01,
           include_zoom_camera=False,
           scale_zoom_camera_to_full_screen=False,
           **kwargs,
   ):
       self.file_name = file_name
       self.svg_config = svg_config
       self.time_per_symbol = time_per_symbol

       super().__init__(
           n_vectors=n_vectors,
           # name_color=name_color,        # 这是个啥
           tex_class=tex_class,
           tex=tex,
           tex_config=tex_config,
           slow_factor=slow_factor,
           parametric_function_step_size=parametric_function_step_size,
           include_zoom_camera=include_zoom_camera,
           scale_zoom_camera_to_full_screen=scale_zoom_camera_to_full_screen,
           **kwargs
       )

   def construct(self):
       self.add_vector_clock()
       if self.tex != None:
           name = self.tex_class(self.tex, **self.tex_config)
       elif self.file_name != None and self.tex == None:
           name = SVGMobject(self.file_name, **self.svg_config)
       # jermain
       max_width = self.camera.frame_width - 2
       max_height = self.camera.frame_height - 2

       # 等比例缩放 jermain
       name.scale_to_fit_width(max_width)
       if name.height > max_height:
           name.scale_to_fit_height(max_height)

       frame = self.camera.frame
       frame.save_state()
       vectors = VGroup(VectorizedPoint())
       circles = VGroup(VectorizedPoint())

       for path in name.family_members_with_points():
           for subpath in path.get_subpaths():
               sp_mob = VMobject()
               sp_mob.set_points(subpath)
               coefs = self.get_coefficients_of_path(sp_mob)
               new_vectors = self.get_rotating_vectors(
                   coefficients=coefs
               )
               new_circles = self.get_circles(new_vectors)
               self.set_decreasing_stroke_widths(new_circles)

               drawn_path = self.get_drawn_path(new_vectors)
               drawn_path.clear_updaters()
               drawn_path.set_style(**self.tex_config)
               drawn_path.set_style(**self.svg_config)

               static_vectors = VMobject().become(new_vectors)
               static_circles = VMobject().become(new_circles)

               self.play(
                   # 替换上一个小路径subpath的向量vectors与圆圈circles
                   Transform(vectors, static_vectors, remover=True),
                   Transform(circles, static_circles, remover=True),
                   # frame.set_height, 1.5 * name.get_height(),
                   # frame.move_to, path,
                   # 修改
                   frame.animate.set_height(1.5 * name.height),
                   frame.animate.move_to(path),
               )

               self.add(new_vectors, new_circles)
               self.vector_clock.set_value(0)
               self.play(
                   Create(drawn_path),
                   rate_func=linear,
                   run_time=self.time_per_symbol
               )

               self.remove(new_vectors, new_circles)
               self.add(static_vectors, static_circles)

               vectors = static_vectors
               circles = static_circles
       self.play(
           FadeOut(vectors),
           FadeOut(circles),
           # Restore(frame) 方法会将场景恢复到指定的帧，这个帧可以是之前保存的场景状态或者之前执行的某个动画的结束状态
           Restore(frame),
           run_time=2
       )
       self.wait(3)

########################################################################################
########################################################################################
########################################################################################

#  ____  _                 _        _____                _
# / ___|(_)_ __ ___  _ __ | | ___  |  ___|__  _   _ _ __(_) ___ _ __
# \___ \| | '_ ` _ \| '_ \| |/ _ \ | |_ / _ \| | | | '__| |/ _ \ '__|
#  ___) | | | | | | | |_) | |  __/ |  _| (_) | |_| | |  | |  __/ |
# |____/|_|_| |_| |_| .__/|_|\___| |_|  \___/ \__,_|_|  |_|\___|_|
#                   |_|
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Using TexMobject or TextMobject
# 1 绘制 Tex、MathTex 或者 Text 对象使用如下的设置
class FourierOfTexSymbol(AbstractFourierOfTexSymbol):
   def __init__(
           self,
           # if start_draw = True the path start to draw
           start_drawn=True,
           # Tex config
           tex_class=MathTex,
           tex=r"\Sigma",
           tex_config={
               "fill_opacity": 0,
               "stroke_width": 1,
               "stroke_color": WHITE
           },
           # Draw config
           drawn_path_color=YELLOW,
           interpolate_config=[0, 1],
           n_vectors=50,
           big_radius=2,
           drawn_path_stroke_width=2,
           center_point=ORIGIN,
           # Duration config
           slow_factor=0.1,
           n_cycles=None,
           run_time=10,
           # colors of circles
           colors=[
               BLUE_D,
               BLUE_C,
               BLUE_E,
               GREY_BROWN,
           ],
           # circles config
           circle_config={
               'stroke_width': 1
           },
           # vector config
           vector_config={
               "buff": 0,
               "max_tip_length_to_length_ratio": 0.2,
               "tip_length": 0.15,
               "max_stroke_width_to_length_ratio": 10,
               "stroke_width": 1.7,
           },
           base_frequency=1,
           # definition of subpaths
           parametric_function_step_size=0.001,
           **kwargs,
   ):

       super().__init__(
           # if start_draw = True the path start to draw
           start_drawn=start_drawn,
           # Tex config
           tex_class=tex_class,
           tex=tex,
           tex_config=tex_config,
           # Draw config
           drawn_path_color=drawn_path_color,
           interpolate_config=interpolate_config,
           n_vectors=n_vectors,
           big_radius=big_radius,
           drawn_path_stroke_width=drawn_path_stroke_width,
           center_point=center_point,
           # Duration config
           slow_factor=slow_factor,
           n_cycles=n_cycles,
           run_time=run_time,
           colors=colors,
           # circles config
           circle_config=circle_config,
           # vector config
           vector_config=vector_config,
           base_frequency=base_frequency,
           # definition of subpaths
           parametric_function_step_size=parametric_function_step_size,
           **kwargs
       )

# Tex examples -------------------------------------------
# # 不同参数设置效果
# n_vectors
# 1.1 绘制 Tex 对象的例子，改变傅里叶级数展开的谐波次数，次数越高，越接近原图
class Tsymbol20vectors(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=50,
           run_time=10,
           tex_class=Text,
           tex="T",
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           run_time=run_time,
           tex_class=tex_class,
           tex=tex,
           **kwargs
       )

class Tsymbol50vectors(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=50,
           run_time=10,    # 10 seconds
           tex_class=Text,
           tex="T",
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           run_time=run_time,
           tex_class=tex_class,
           tex=tex,
           **kwargs
       )

class Tsymbol150vectors(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=150,
           run_time=10,    # 10 seconds
           tex_class=Text,
           tex="T",
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           run_time=run_time,
           tex_class=tex_class,
           tex=tex,
           **kwargs
       )

# 求和符号
class SigmaSymbol150vectors(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=150,
           run_time=10,
           tex_class=Tex, # <-------- Default
           tex=r"$\Sigma$",
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           run_time=run_time,
           tex_class=tex_class,
           tex=tex,
           **kwargs
       )


# slow_factor
# 1.2 Tex 对象：改变 slow_factor 速度因子
# # 完成绘制一周轮廓的时间为 1/slow_factor
# # slow_factor=1 指1s完成一周轮廓的绘制，数字越小，越慢
class SlowFactor0_1(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=30,
           run_time=7,
           tex_class=Tex,
           tex=r"$\Sigma$",
           slow_factor=0.1,
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           run_time=run_time,
           tex_class=tex_class,
           tex=tex,
           slow_factor=slow_factor,
           **kwargs
       )

class SlowFactor0_3(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=30,
           run_time=7,
           tex_class=Tex,
           tex="\\Sigma",
           slow_factor=0.3,
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           run_time=run_time,
           tex_class=tex_class,
           tex=tex,
           slow_factor=slow_factor,
           **kwargs
       )

class SlowFactor0_5(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=30,
           run_time=7,
           tex_class=Tex,
           tex=r"$\Sigma$",
           slow_factor=0.5,
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           run_time=run_time,
           tex_class=tex_class,
           tex=tex,
           slow_factor=slow_factor,
           **kwargs
       )

# start_drawn
# 1.3 Tex 对象：改变 start_drawn=True or False
# # 是否需要提前绘制好合成的路径
class StartDrawTrue(FourierOfTexSymbol):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=30,
           run_time=7,
           tex="\\tau",
           start_drawn=True,   # <------------------ Default
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           run_time=run_time,
           tex=tex,
           start_drawn=start_drawn,
           **kwargs
       )

class StartDrawFalse(FourierOfTexSymbol):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=30,
           run_time=7,
           tex="\\tau",
           start_drawn=False,   # <------------------
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           run_time=run_time,
           tex=tex,
           start_drawn=start_drawn,
           **kwargs
       )

# interpolate_config
# 1.4 Tex 对象，改变 interpolate_config
# # 设置当前时刻绘制路径和之前时刻绘制曲线的路径的粗细比例
class InterpolateConfig0to1(FourierOfTexSymbol):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=30,
           run_time=15,
           tex="\\tau",
           interpolate_config=[0, 1],   # <---------- Default
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           run_time=run_time,
           tex=tex,
           interpolate_config=interpolate_config,
           **kwargs
       )

class InterpolateConfig0_3to_1(FourierOfTexSymbol):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=30,
           run_time=15,
           tex="\\tau",
           interpolate_config=[0.3, 1],   # <----------
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           run_time=run_time,
           tex=tex,
           interpolate_config=interpolate_config,
           **kwargs
       )

class InterpolateConfig1_to_1(FourierOfTexSymbol):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=30,
           run_time=15,
           tex="\\tau",
           interpolate_config=[1, 1],   # <---------- # always write
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           run_time=run_time,
           tex=tex,
           interpolate_config=interpolate_config,
           **kwargs
       )

# n_cycles vs run_time
# 1.5 n_cycles 和 run_time 是定义运行时间的两种方式，n_cycles有较高的优先级
# # 默认 n_cycles=None, run_time=10, 同时设置的话默认使用 n_cycles
class NCyclesVsRunTime(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=30,
           n_cycles=3,
           tex="\\tau",
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           n_cycles=n_cycles,
           tex=tex,
           **kwargs
       )

# wait_before_start
# 1.6 绘制前的等待时间
class WaitBeforeStart(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=30,
           n_cycles=1,
           tex="\\tau",
           wait_before_start=2,
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           n_cycles=n_cycles,
           tex=tex,
           wait_before_start=wait_before_start,
           **kwargs
       )

# center_point
class CenterPoint(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=30,
           n_cycles=1,
           tex="\\tau",
           center_point=RIGHT*5,
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           n_cycles=n_cycles,
           tex=tex,
           center_point=center_point,
           **kwargs
       )

# path_custom_position
class CustomPosition(FourierOfTexSymbol):
   def __init__(
           self,
           n_vectors=30,
           n_cycles=1,
           tex="\\tau",
           center_point=RIGHT*3,
           path_custom_position=lambda mob: mob.to_edge(LEFT),
           circle_config={
               'stroke_opacity': 0
           },
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           n_cycles=n_cycles,
           tex=tex,
           center_point=center_point,
           path_custom_position=path_custom_position,
           circle_config=circle_config,
           **kwargs
       )


# o x o x o x o x o x o x o x o x o x o x o x o x o x o x o x o x o x o x o x o x
# SVG
# 2 SVG 图像绘制
class FourierFromSVG(AbstractFourierFromSVG):
   CONFIG = {
       # if start_draw = True the path start to draw
       "start_drawn": True,
       # SVG file name
       "file_name": None,
       "svg_config": {
           "fill_opacity": 0,
           "stroke_color": WHITE,
           "stroke_width": 1,
           "height": 7
       },
       # Draw config
       "drawn_path_color": YELLOW,
       "interpolate_config": [0, 1],
       "n_vectors": 50,
       "big_radius": 2,
       "drawn_path_stroke_width": 2,
       "center_point": ORIGIN,
       # Duration config
       "slow_factor": 0.1,
       "n_cycles": None,
       "run_time": 10,
       # colors of circles
       "colors": [
           BLUE_D,
           BLUE_C,
           BLUE_E,
           GREY_BROWN,
       ],
       # circles config
       "circle_config": {
           "stroke_width": 1,
       },
       # vector config
       "vector_config": {
           "buff": 0,
           "max_tip_length_to_length_ratio": 0.25,
           "tip_length": 0.15,
           "max_stroke_width_to_length_ratio": 10,
           "stroke_width": 1.7,
       },
       "base_frequency": 1,
       # definition of subpaths
       "parametric_function_step_size": 0.001,
   }
   def __init__(
           self,
           # if start_draw = True the path start to draw
           start_drawn=True,
           # SVG file name
           file_name=None,
           svg_config={
               "fill_opacity": 0,
               "stroke_color": WHITE,
               "stroke_width": 1,
               "height": 7
           },
           # Draw config
           drawn_path_color=YELLOW,
           interpolate_config=[0, 1],
           n_vectors=50,
           big_radius=2,
           drawn_path_stroke_width=2,
           center_point=ORIGIN,
           # Duration config
           slow_factor=0.1,
           n_cycles=None,
           run_time=10,
           # colors of circles
           colors=[
               BLUE_D,
               BLUE_C,
               BLUE_E,
               GREY_BROWN,
           ],
           # circles config
           circle_config={
               'stroke_width': 1
           },
           # vector config
           vector_config={
               "buff": 0,
               "max_tip_length_to_length_ratio": 0.25,
               "tip_length": 0.15,
               "max_stroke_width_to_length_ratio": 10,
               "stroke_width": 1.7,
           },
           base_frequency=1,
           # definition of subpaths
           parametric_function_step_size=0.001,
           **kwargs,
   ):
       super().__init__(
           start_drawn=start_drawn,
           file_name=file_name,
           svg_config=svg_config,
           drawn_path_color=drawn_path_color,
           interpolate_config=interpolate_config,
           n_vectors=n_vectors,
           big_radius=big_radius,
           drawn_path_stroke_width=drawn_path_stroke_width,
           center_point=center_point,
           slow_factor=slow_factor,
           n_cycles=n_cycles,
           run_time=run_time,
           colors=colors,
           circle_config=circle_config,
           vector_config=vector_config,
           base_frequency=base_frequency,
           parametric_function_step_size=parametric_function_step_size,
           **kwargs
       )

# file_name
class SVGDefault(FourierFromSVG):
   def __init__(
           self,
           n_vectors=100,
           n_cycles=1,
           file_name="c_clef", #
        #    file_name="./assets/svg_images/c_clef.svg",
           **kwargs,
   ):
       super().__init__(
           n_vectors=n_vectors,
           n_cycles=n_cycles,
           file_name=file_name,
           **kwargs
       )

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# __        ___ _   _
# \ \      / (_) |_| |__
#  \ \ /\ / /| | __| '_ \
#   \ V  V / | | |_| | | |
#    \_/\_/  |_|\__|_| |_|
#  _____                              _
# |__  /___   ___  _ __ ___   ___  __| |   ___ __ _ _ __ ___   ___ _ __ __ _
#   / // _ \ / _ \| '_ ` _ \ / _ \/ _` |  / __/ _` | '_ ` _ \ / _ \ '__/ _` |
#  / /| (_) | (_) | | | | | |  __/ (_| | | (_| (_| | | | | | |  __/ | | (_| |
# /____\___/ \___/|_| |_| |_|\___|\__,_|  \___\__,_|_| |_| |_|\___|_|  \__,_|
# ---------------------------------------------------------------------------------------
# The following works in both Tex and SVG
# ---------------------------------------
#      ---------------------------
#          ----------------

# How activate it
# 3 使用近景图，局部放大
class ZoomedActivate(FourierFromSVG):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=50,
           n_cycles=1,
           file_name="c_clef",
        #    file_name="./assets/svg_images/c_clef.svg",
           include_zoom_camera=True,
           zoom_position=lambda zc: zc.to_corner(DR),
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           n_cycles=n_cycles,
           file_name=file_name,
           include_zoom_camera=include_zoom_camera,
           zoom_position=zoom_position,
           **kwargs
       )


# Zoomed camera: Moving camera
# Zoomed display: Static camera
# More info: https://github.com/Elteoremadebeethoven/AnimationsWithManim/blob/master/English/extra/faqs/faqs.md#zoomed-scene-example
class ZoomedConfig(FourierFromSVG):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=150,
           n_cycles=1,
           file_name="c_clef",
        #    file_name="./assets/svg_images/c_clef.svg",
           path_custom_position=lambda path: path.shift(LEFT*2),
           center_point=LEFT*2,
           circle_config={
               "stroke_width": 0.5,
               "stroke_opacity": 0.2,
           },
           # Zoom config
           include_zoom_camera=True,
           zoom_position=lambda zc: zc.to_edge(RIGHT).set_y(0),
           zoom_factor=0.5,
           zoomed_display_height=5,
           zoomed_display_width=5,
           zoomed_camera_config={
               "default_frame_stroke_width": 3,
               "cairo_line_width_multiple": 0.05,
               # What is cairo_line_width_multiple?
               # See here: https://stackoverflow.com/questions/60765530/manim-zoom-not-preserving-line-thickness
           },
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           n_cycles=n_cycles,
           file_name=file_name,
           path_custom_position=path_custom_position,
           center_point=center_point,
           circle_config=circle_config,
           include_zoom_camera=include_zoom_camera,
           zoom_position=zoom_position,
           zoom_factor=zoom_factor,
           zoomed_display_height=zoomed_display_height,
           zoomed_display_width=zoomed_display_width,
           zoomed_camera_config=zoomed_camera_config,
           **kwargs
       )

# Move Zoomed display to full screen
class ZoomedDisplayToFullScreen(FourierOfTexSymbol):
   def __init__(
           self,
           slow_factor=0.05,
           n_vectors=30,
           run_time=16,
           tex="\\tau",
           # Zoom config
           include_zoom_camera=True,
           zoom_position=lambda zc: zc.to_corner(DR),
           # Zoomed display to Full screen config
           scale_zoom_camera_to_full_screen=True,
           scale_zoom_camera_to_full_screen_at=4,
           zoom_camera_to_full_screen_config={
               "run_time": 4,
               "func": smooth,
               "velocity_factor": 1
           },
           **kwargs,
   ):
       super().__init__(
           slow_factor=slow_factor,
           n_vectors=n_vectors,
           run_time=run_time,
           tex=tex,
           # Zoom config
           include_zoom_camera=include_zoom_camera,
           zoom_position=zoom_position,
           # Zoomed display to Full screen config
           scale_zoom_camera_to_full_screen=scale_zoom_camera_to_full_screen,
           scale_zoom_camera_to_full_screen_at=scale_zoom_camera_to_full_screen_at,
           zoom_camera_to_full_screen_config=zoom_camera_to_full_screen_config,
           **kwargs
       )

class ZoomedDisplayToFullScreenWithRestore(ZoomedDisplayToFullScreen):
   def __init__(
           self,
           run_time=20,
           zoom_camera_to_full_screen_config={
               "run_time": 12,
               "func": lambda t: there_and_back_with_pause(t, 1/10),
               # learn more: manimlib/utils/rate_functions.py
               "velocity_factor": 1,
           },
           **kwargs
   ):
       super().__init__(
           run_time=run_time,
           zoom_camera_to_full_screen_config=zoom_camera_to_full_screen_config,
           **kwargs
       )


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

#  ____                                   _   _
# |  _ \ _ __ __ ___      __  _ __   __ _| |_| |__
# | | | | '__/ _` \ \ /\ / / | '_ \ / _` | __| '_ \
# | |_| | | | (_| |\ V  V /  | |_) | (_| | |_| | | |
# |____/|_|  \__,_| \_/\_/   | .__/ \__,_|\__|_| |_|
#                            |_|

# //////////////////////////////////////////////////////////////////////////////////////////

class FourierOfPathsTB(FourierOfPaths):
   def __init__(
           self,
           n_vectors=100,
           tex_class=Text,
           tex="TB",
           tex_config={
               "stroke_color": RED,
           },
           time_per_symbol=5,
           slow_factor=1 / 5,
           **kwargs
   ):
       super().__init__(
           n_vectors=n_vectors,
           tex_class=tex_class,
           tex=tex,
           tex_config=tex_config,
           time_per_symbol=time_per_symbol,
           slow_factor=slow_factor,
           **kwargs
       )


# Convert objects to paths
# Inkscape example: [ToolBar] Path > Object to path
class FourierOfPathsSVG(FourierOfPaths):
   def __init__(
           self,
           n_vectors=80,
           file_name="./assets/svg_images/music_symbols.svg",
           svg_config={
               "stroke_color": RED,
           },
           time_per_symbol=4,      # 每个svg路径绘制的时间
           # 为了上一行设置的时间内将图像绘制一周，设置slow_factor如下
           slow_factor=1 / 4,
           **kwargs
   ):
       super().__init__(
           n_vectors=n_vectors,
           file_name=file_name,
           svg_config=svg_config,
           time_per_symbol=time_per_symbol,
           slow_factor=slow_factor,
           **kwargs
       )


# 绘制蓝兔
class FourierOfPathsMySVG(FourierOfPaths):
   def __init__(
           self,
           n_vectors=80,
           file_name="./assets/svg_images/头像.svg",
           svg_config={
               "stroke_color": BLUE,
           },
           time_per_symbol=4,      # 每个svg路径绘制的时间
           # 为了上一行设置的时间内将图像绘制一周，设置slow_factor如下
           slow_factor=1 / 4,
           **kwargs
   ):
       super().__init__(
           n_vectors=n_vectors,
           file_name=file_name,
           svg_config=svg_config,
           time_per_symbol=time_per_symbol,
           slow_factor=slow_factor,
           **kwargs
       )


# //////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////

#   ____          _
#  / ___|   _ ___| |_ ___  _ __ ___
# | |  | | | / __| __/ _ \| '_ ` _ \
# | |__| |_| \__ \ || (_) | | | | | |
#  \____\__,_|___/\__\___/|_| |_| |_|

#     _          _                 _   _
#    / \   _ __ (_)_ __ ___   __ _| |_(_) ___  _ __  ___
#   / _ \ | '_ \| | '_ ` _ \ / _` | __| |/ _ \| '_ \/ __|
#  / ___ \| | | | | | | | | | (_| | |_| | (_) | | | \__ \
# /_/   \_\_| |_|_|_| |_| |_|\__,_|\__|_|\___/|_| |_|___/
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

class CustomAnimationExample(FourierCirclesScene):
   def __init__(
           self,
           n_vectors=200,
           slow_factor=0.12,
           fourier_symbol_config={
               "stroke_width": 0,
               "fill_opacity": 0,
               "height": 2,
               "fill_color": WHITE
           },
           circle_config={
               "stroke_width": 1,
               "stroke_opacity": 0.3,
           },
           **kwargs
   ):
       self.fourier_symbol_config = fourier_symbol_config
       super().__init__(
           n_vectors=n_vectors,
           slow_factor=slow_factor,
           circle_config=circle_config,
           **kwargs)

   def construct(self):
       
       t_symbol = Text("T", **self.fourier_symbol_config)
    #    c_clef_symbol = SVGMobject("./assets/svg_images/c_clef.svg", **self.fourier_symbol_config)
    #    c_clef_symbol.match_height(t_symbol)
    #    # set gradient
    #    for mob in [t_symbol,c_clef_symbol]:
    #        # mob.set_sheen(0,UP)
    #        # 可以更改为
    #        # set_sheen_direction(UP)：这个方法用于设置对象的光泽方向。参数 UP 表示光泽的方向向上。光泽效果可以让对象看起来更加立体和有光泽感。
    #        mob.set_sheen_direction(UP)
    #        # set_color(color=[BLACK, GRAY, WHITE])：这个方法用于设置对象的颜色。参数 color 是一个列表，表示对象的渐变色。
    #        # 在这里，使用了一个包含三个颜色的列表，表示对象的颜色从黑色（BLACK）渐变到灰色（GRAY），再渐变到白色（WHITE）。这样可以让对象的颜色呈现出渐变的效果。
    #        mob.set_color(color=[BLACK,GRAY,WHITE])
    #        # 修改 -> jermain
    #        # mob.set_color_by_gradient(BLACK, GRAY, WHITE)
    #    group = VGroup(t_symbol,c_clef_symbol).arrange(RIGHT,buff=0.1)

       # set paths
       # 提取svg文件的路径
       path1 = t_symbol.family_members_with_points()[0]
    #    path2 = c_clef_symbol.family_members_with_points()[0]

       # path 1 config
       # 修改 -> jermain
       # 求出路径path的fourier系数
    #    coefs1 = self.get_coefficients_of_path(path1)
    #    print(len(coefs1))
    #    print(coefs1)
       # 根据系数生成对应的向量vector
    #    vectors1 = self.get_rotating_vectors(coefficients=coefs1)
    #    vectors1_to_fade = self.get_rotating_vectors(coefficients=coefs1)
       # 利用向量生成对应的圆circle
    #    circles1 = self.get_circles(vectors1)
    #    circles1_to_fade = self.get_circles(vectors1_to_fade)
       # 根据生成的向量生成末端的绘画轨迹
    #    drawn_path1 = self.get_drawn_path(vectors1)
    #    print(drawn_path1)
       # path 2 config
       # 求出路径path的fourier系数
       #coefs2 = self.get_coefficients_of_path(path2)
       #print("coefs2:")
       #print(coefs2)
       #print(len(coefs2))

    #    audio = AudioSegment.from_mp3("兰亭序.mp3")
       audio = AudioSegment.from_mp3("宝岛.mp3")
       data = np.array(audio.get_array_of_samples())
       sample_rate = audio.frame_rate
       data_half = data[::2]
    #    fft_result = fft(data_half,n=44100*254)
       # fft结果
       fft_result = fft(data_half)
       print(len(fft_result))
       # 取fft生成序列长度的一半
       N = len(fft_result)//2
       print(N)
       # 取fft生成序列的一半（原本是打算去正频率，0:N的话应该是只取负频率，到时候看看正负频率之间的区别）
       coefs2_fft = fft_result[0:N]
       numpy_array = np.array(coefs2_fft)
       # 对fft结果除以N，其中0频率（）直流分量除以N，其他频率分量除以N/2
       new_array = np.concatenate(([numpy_array[0]/N], numpy_array[1:-1]*2/N, [numpy_array[-1]/N]))
       print("new_array length")
       print(len(new_array))
       # 因为使用所有样本点进行绘画需要大量的渲染时间，这里选择对频率分量进行抽样
       # 又因为采样频率是44100，所以每一个点之间的怕频率间隔为1/44100
       # 每隔44100个点相差1Hz，这里选这每100Hz选用一个频率分量（这个分量取值是这个100个Hz点之间的均值）
       #TODO 这里对频率分量的处理可以自行修改，寻找合适的处理方法可能得到更好的绘画效果
       coefs2 = np.array([np.mean(new_array[i:i+N//44100*100]) for i in range(0, N, N//44100*100)])
    #    coefs2 = np.array([np.mean(new_array[i:i+N//44100]) for i in range(0, N, N//44100)])
       print("coefs2 length")
       print(len(coefs2))

       #!对于不同的歌曲有不同的赋值分量，直接画出来可能会出现图像过大和过小的现象，需要我们对频率的复制分量进行适当的放大
       #coefs2 = coefs2_fft[0:200]
       #四季专用
    #    coefs2 = [x * 220 for x in coefs2]
    #    coefs2 = [x * 20 for x in coefs2]
       coefs2 = [x * 60 for x in coefs2]
    #    coefs2 = [x * 0.002 for x in coefs2]
    #    coefs2 = [[i] for i in coefs2_fft_single]
       print("coefs2 length")
       print(len(coefs2))
    #    coefs2 = coefs2[0:len(coefs2)//5]
    #    print("coefs2 true length")
    #    print(len(coefs2))

       # 根据系数生成对应的向量vector
       vectors2 = self.get_rotating_vectors(coefficients=coefs2)
       vectors2_to_fade = self.get_rotating_vectors(coefficients=coefs2)
       # 利用向量生成对应的圆circle
       circles2 = self.get_circles(vectors2)
       circles2_to_fade = self.get_circles(vectors2_to_fade)
       # 根据生成的向量生成末端的绘画轨迹
       drawn_path2 = self.get_drawn_path(vectors2)
       drawn_path2_py = self.get_drawn_path_py(vectors2)
    #    print(drawn_path2)

       # text definition
    #    text = Text("Thanks for watch!")
    #    text.scale(1.5)
    #    text.next_to(group,DOWN)

       # all elements toget
    #    all_mobs = VGroup(group, text)

       # set mobs to remove
    #    vectors1_to_fade.clear_updaters()
    #    circles1_to_fade.clear_updaters()

       vectors2_to_fade.clear_updaters()
       circles2_to_fade.clear_updaters()

    #    self.play(
    #        *[
    #            GrowArrow(arrow)
    #            for vg in [vectors1_to_fade, vectors2_to_fade]
    #            for arrow in vg
    #        ],
    #        *[
    #            Create(circle)
    #            for cg in [circles1_to_fade, circles2_to_fade]
    #            for circle in cg
    #        ],
    #        run_time=2.5,
    #    )
       self.play(
           *[
               GrowArrow(arrow)
               for vg in [vectors2_to_fade]
               for arrow in vg
           ],
           *[
               Create(circle)
               for cg in [circles2_to_fade]
               for circle in cg
           ],
           run_time=2.5,
       )
       self.remove(
        #    *vectors1_to_fade,
        #    *circles1_to_fade,
           *vectors2_to_fade,
           *circles2_to_fade,
       )
       self.add(
        #    vectors1,
        #    circles1,
        #    drawn_path1.set_color(RED),
           vectors2,
           circles2,
           drawn_path2.set_color(RED),
       )

       self.add_vector_clock()

       # wait one cycle
       self.wait(1 / self.slow_factor)
       self.wait(15)


    #    self.remove(
    #        vectors2,
    #        circles2,
    #        drawn_path2.set_color(RED),)
       
    #    self.add(

    #     #    vectors1,
    #     #    circles1,
    #     #    drawn_path1.set_color(RED),
    #        vectors2,
    #        circles2,
    #        drawn_path2.set_color(BLUE),
    #    )


    #    self.bring_to_back(t_symbol, c_clef_symbol)
    #    self.play(
    #        t_symbol.animate.set_fill(color=None, opacity=1),
    #        c_clef_symbol.animate.set_fill(color=None, opacity=1),
    #        run_time=3
    #    )
    #    self.wait()
       # move camera
    #    self.play(
    #        # self.camera_frame.set_height, all_mobs.get_height()*1.2,
    #        # self.camera_frame.move_to, all_mobs.get_center()
    #        # 修改 -> jermain
    #     #    self.camera.frame.animate.set_height(all_mobs.get_height()*4),
    #     #    self.camera.frame.animate.move_to(all_mobs.get_center())
    #         # self.camera.frame.animate.set_height(circles2.get_height()*4),
    #         # self.camera.frame.animate.move_to(circles2.get_center())
    #    )
       self.wait(0.5)
    #    self.play(
    #        Write(text)
    #    )
       self.wait(15)

# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-

