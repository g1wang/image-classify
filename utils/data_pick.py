from PIL import Image
import os
import shutil
import uuid

src_dir_path = 'G:\\MF-Images'
dest_dir_path = 'D:\\dataset\\image-classify\\mf'

if not os.path.exists(dest_dir_path):
    os.mkdir(dest_dir_path)

for fpathe, dirs, fs in os.walk(src_dir_path):
    for f in fs:
        childPath = os.path.join(fpathe, f)
        if os.path.isdir(childPath):
            #print(childPath)
            pass
        else:
            try:
                img = Image.open(childPath).convert('RGB')
                shutil.move(os.path.join(childPath), os.path.join(dest_dir_path, str(uuid.uuid1()) + '.jpg'))
            except:
                 pass