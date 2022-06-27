import os
import os.path
import shutil
path = "D:/USER_NAME/DESKTOP/数据集/RoLabelImg_Transform/txt/"

filelist = os.listdir(path)
for file in filelist:
    Olddir = os.path.join(path, file)
    if os.path.isdir(Olddir):
        continue
    filename = os.path.splitext(file)[0]
    filetype = os.path.splitext(file)[1]
    filename = int(filename)
    for i in range (80):
        newfilename = filename + i
        newfilepath = path +'new/' + format(newfilename, '0>6d') + filetype
        shutil.copyfile(path+file,newfilepath)
