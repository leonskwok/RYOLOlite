import os
import os.path
import shutil
# path = "D:/USER_NAME/DESKTOP/数据集/img/"
path = "D:/USER_NAME/DESKTOP/数据集/数据处理/before_label/"
filelist = os.listdir(path)
for i in range(len(filelist)):
    Olddir = os.path.join(path, filelist[i])
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(filelist[i])[0]
    filetype = os.path.splitext(filelist[i])[1]
    idx = int(filename)
    if idx%80==0:
        newfilepath = path + 'new/' + str(int(idx/80)) + filetype
        shutil.copyfile(path+filelist[i], newfilepath)
