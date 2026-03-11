import os
import random
import sys

def read_files(path):
    train_set = dataset_path +'/train.txt'
    img_list = []
    with open(train_set) as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            img_file, _ = line.split(' ')
            img_list.append(img_file)
    return img_list

def scan_files(path):
    return os.listdir(path)


def random_select(img_list, num = 50):
    total_num = len(img_list)
    print('found files total: {}'.format(total_num))
    assert(total_num >= num)
    res_list = {}
    while len(res_list) < num:
        n = random.randint(0, total_num - 1)
        if n not in res_list:
            res_list[n] = img_list[n]
    return res_list

if __name__ == "__main__":
    dataset_path = sys.argv[1]
    images_num = eval(sys.argv[2])

    img_list = scan_files(dataset_path)
    res_list = random_select(img_list, images_num)

    fo = open('images.list', 'w+')
    for i in res_list:
        line = os.path.join(dataset_path, res_list[i])
        fo.write(line + '\n')
    fo.close()