import os

# 想要更改图片所在的根目录
rootdir = "./data/VOCdevkit/VOC2012/JPEGImages"
# 获取目录下文件名清单
files = os.listdir(rootdir)
print(os.getcwd())
os.chdir("./data/VOCdevkit/VOC2012/JPEGImages")  # 修改工作路径
# 对文件名清单里的每一个文件名进行处理
for filename in files:
    print(filename)
    portion = os.path.splitext(filename)  # portion为名称和后缀分离后的列表   #os.path.splitext()将文件名和扩展名分开
    if portion[1] == ".JPG":  # 如果为tiff则更改名字
        newname = portion[0] + ".jpg"  # 要改的新后缀  #改好的新名字
        print(filename)  # 打印出要更改的文件名

        os.rename(filename, newname)  # 在工作路径下对文件名重新命名