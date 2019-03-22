from PIL import Image
import os
import shutil

class_0_dir = 'D:\\dataset\\image-classify\\hupu'

datalist = os.listdir(class_0_dir)
for list in datalist:
    path = os.path.join(class_0_dir, list)
    try:
        img = Image.open(path).convert('RGB')
    except:
        try:
            shutil.move(os.path.join(path), os.path.join('D:\\dataset\\image-classify\\failimg', list))
        except:
            pass
