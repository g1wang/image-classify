import os
import shutil
import uuid

src_dir_path = 'G:\\百度图片_书籍'
dest_dir_path = 'D:\\dataset\\image-classify\\书籍'
if not os.path.exists(dest_dir_path):
    os.mkdir(dest_dir_path)


for fpathe, dirs, fs in os.walk(src_dir_path):
    for f in fs:
        childPath = os.path.join(fpathe, f)
        if os.path.isdir(childPath):
            print(childPath)
        shutil.move(childPath, os.path.join(dest_dir_path, str(uuid.uuid1()) + '.jpg'))
