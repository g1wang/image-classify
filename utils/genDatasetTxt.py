import os

from PIL import Image

class_0_dir = ''
class_1_dir = ''
# class_0_dir = 'D:\dataset\\porn-pure\\train\\normal'
# class_1_dir = 'D:\dataset\\porn-pure\\train\\porn'
# filepath = './porn-train.txt'

# class_0_dir = 'D:\dataset\\porn-pure\\validation\\normal'
# filepath = './normal-validation.txt'

# class_1_dir = 'D:\dataset\\porn-pure\\validation\\porn'
# filepath = './porn-validation.txt'

# class_0_dir = 'D:\dataset\\porn-pure\\test\\normal'
# filepath = './normal-test.txt'

# class_1_dir = 'D:\dataset\\porn-pure\\test\\porn'
# filepath = './porn-test.txt'

# class_1_dir = 'D:\dataset\\porn-pure\\train\\porn'
# filepath = './porn-train-porn.txt'

# class_0_dir = 'D:\\dataset\\porn-pure\\train\\normal'
# filepath = './porn-train-normal.txt'

class_1_dir = 'D:\\dataset\\downSamples-part1'
filepath = './porn-test.txt'



lines = []
if class_0_dir != '':
    catslist = os.listdir(class_0_dir)
    for list in catslist:
        path = ''
        try:
            path = os.path.join(class_0_dir, list)
            # im = Image.open(path)
        except Exception:
            print(path + ' load err')
            continue
        print(path)
        line = path + ' 0\n'
        lines.append(line)

if class_1_dir != '':
    dogslist = os.listdir(class_1_dir)
    for list in dogslist:
        path = ''
        try:
            path = os.path.join(class_1_dir, list)
            # im = Image.open(path)
        except Exception:
            print(path + ' load err')
            continue
        print(path)
        line = path + ' 1\n'
        lines.append(line)
file = open(filepath, 'w+')
file.writelines(lines)
