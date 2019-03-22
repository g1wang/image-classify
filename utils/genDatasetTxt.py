import os

class_0_dir = ''
class_1_dir = ''
class_2_dir=''
filepath=''

# train
#
# class_0_dir = 'D:\\dataset\\image-classify\\train\\n'
# class_1_dir = 'D:\\dataset\\image-classify\\train\\p'
# class_2_dir = 'D:\\dataset\\image-classify\\train\\s'
# filepath = '../datalist/train.txt'

#validation

# class_0_dir = 'd:\dataset\\image-classify\\validation\\n'
# filepath = '../datalist/n-val.txt'

# class_1_dir = 'D:\dataset\\image-classify\\validation\\p'
# filepath = '../datalist/p-val.txt'
#
# class_2_dir = 'D:\dataset\\image-classify\\validation\\s'
# filepath = '../datalist/s-val.txt'

#test

class_0_dir = 'D:\dataset\\image-classify\\hupu'
filepath = '../datalist/classify.txt'

# class_2_dir = 'D:\dataset\\image-classify\\testout\\s'
# filepath = '../datalist/s.txt'


# class_2_dir = 'D:\dataset\\image-classify\\s-part1'
# filepath = '../datalist/s-test.txt'


# 过滤训练集

# class_0_dir = 'D:\\dataset\\image-classify\\train\\n'
# filepath = '../datalist/n-train.txt'

# class_1_dir = 'D:\dataset\\image-classify\\train\\p'
# filepath = '../datalist/p-train.txt'

# class_2_dir = 'D:\\dataset\\image-classify\\train\\s'
# filepath = '../datalist/s-train.txt'





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
if class_2_dir != '':
    dogslist = os.listdir(class_2_dir)
    for list in dogslist:
        path = ''
        try:
            path = os.path.join(class_2_dir, list)
            # im = Image.open(path)
        except Exception:
            print(path + ' load err')
            continue
        print(path)
        line = path + ' 2\n'
        lines.append(line)
file = open(filepath, 'w+')
file.writelines(lines)
