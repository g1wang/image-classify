import os
import shutil
import os

src_dir_path = 'D:\\dataset\\porn\\train\\porn1'
dest_dir_path = 'D:\\dataset\\porn\\train\\porn'
if not os.path.exists(dest_dir_path):
    os.mkdir(dest_dir_path)


i = 108894 #n
#i = 10064282 #p
for fpathe, dirs, fs in os.walk(src_dir_path):
    for f in fs:
        i += 1
        childPath = os.path.join(fpathe, f)
        if os.path.isdir(childPath):
            print(childPath)
        shutil.copy(childPath, os.path.join(dest_dir_path, str(i) + '.jpg'))

print(i)