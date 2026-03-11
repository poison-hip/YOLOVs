import cv2
import numpy as np
import sys, os
import shutil  

def read_files(list_file):
    img_list = []
    with open(list_file) as f:
        img_list = f.read().splitlines()
    return img_list

def reader(img_path, input_size=(416, 416)):
    input = {}
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    scale_factor = [input_size[0] / img.shape[0], input_size[1] / img.shape[1]]
    factor = np.array(scale_factor, dtype=np.float32)
    input['scale_factor'] = factor.reshape((1, 2))
    img = cv2.resize(img, input_size, interpolation=2)
    
    mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
    std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
    img = img / 255
    img -= mean
    img /= std
    img = img.astype(np.float32, copy=False)

    h, w, _ = img.shape
    input['im_shape'] = np.array([[h, w]], dtype=np.int32)

    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :]
    input['image'] = img
    
    return input 

if __name__ == "__main__":
    model_dir = sys.argv[1]
    input_size = eval(sys.argv[2])
    #files_list = sys.argv[3]
    images_dir = sys.argv[3]

    #img_list = read_files(files_list)
    img_list = os.listdir(images_dir)

    for i in range(len(img_list)):
        if img_list[i].endswith('.jpg') or img_list[i].endswith('.png'):
            print(img_list[i])
        else:
            img_list.remove(img_list[i])

    print('found images: {}'.format(len(img_list)))
    model_dir = os.path.join(model_dir, 'output')

    feeds = ['image', 'im_shape', 'scale_factor']
    for i in feeds:
        dir = os.path.join(model_dir, i)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    for i in range(len(img_list)):
        #print('i = {}'.format(i))
        name = img_list[i].split('/')[-1]
        image = os.path.join(images_dir, img_list[i])

        input = reader(image, input_size)
        for j in feeds:
            npy_file = os.path.join(model_dir, j, str(i) + '.npy')
            np.save(npy_file, input[j])
    print('save npy file to: {}'.format(model_dir))
