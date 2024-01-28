from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker
import os
path = './test'
# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
with open('niqe.txt', 'r') as f:
    lst=f.readlines()
    for i in range(len(lst)):
        lst[i]=lst[i][:-1]
    c = (
        Bar()
        .add_xaxis(image_files)
        .add_yaxis("NIQE",lst,category_gap="80%")
        .set_global_opts(title_opts=opts.TitleOpts(title="NIQE指标", subtitle="NIQE值"))            .render("bar_niqe.html")
    )


