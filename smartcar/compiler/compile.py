# -*- coding: utf-8 -*-
import os, sys
import subprocess
import json

PPNC_HOME = os.getenv('PPNC_HOME')
project_path = PPNC_HOME + "/scripts"
build_path = os.getcwd()


def update_io_config(config_file, shape):
    io_config = {}
    with open(config_file, 'r') as f:
        io_config = json.load(f)
        for i in range(len(io_config)):
            if io_config[i]['name'] == 'image':
                io_config[i]['shape'] = [1, 3, *eval(shape)]
                break
        f.close()
    with open(config_file, 'w+') as f:
        json.dump(io_config, f, indent=4)
        f.close()

def update_quantize_config(config_file, path):
    config = {}
    with open(config_file, 'r') as f:
        config = json.load(f)
        for i in range(len(config)):
            name = config[i]['name']
            config[i]['image_path'] = os.path.join(path, 'output', name)
    f.close()
    with open(config_file, 'w+') as f:
        json.dump(config, f, indent=4)
        f.close()

def package(model_dir, package_name = 'output', split = False):
    ret = subprocess.run(['mkdir -p ' + package_name], shell=True, cwd=build_path)
    assert(ret.returncode == 0)

    io_config = os.path.join(project_path, 'res/io_paddle.json ')
    if split:
        io_config = os.path.join(model_dir, 'output/io_paddle.json ')
    print(os.path.abspath(__file__))
    cmd = [
        'cp -f',
        'deploy_paddle.params',
        'deploy_paddle.so',
        'deploy_paddle.ro',
        'deploy_paddle.tar',
        io_config
    ]
    if split:
        cmd.append(os.path.join(model_dir, 'output/nms.pdmodel'))
        cmd.append(os.path.join(model_dir, 'output/post.onnx'))
        cmd.append(package_name)
    shell_cmd = ' '.join(cmd)

    ret = subprocess.run(shell_cmd, shell=True, cwd=build_path)
    assert(ret.returncode == 0)

    ret = subprocess.run([
        'zip -q -r ' +
        package_name + '.zip ' +
        package_name
    ], shell=True, cwd=build_path)
    assert(ret.returncode == 0)
    print('export:{}/{}.zip'.format(build_path, package_name))


if __name__ == "__main__":
    print('-----------------------------------------------')
    print('---------------begin--------------')
    print('-----------------------------------------------')

    assert(len(PPNC_HOME) != 0)
    print('PPNC_HOME:{}'.format(PPNC_HOME))
    subprocess.run(['mkdir -p build'], shell=True)

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = json.load(f)

    model_dir = config['model_dir']
    shape = config['shape']
    split = config['split']
    io_config = os.path.join(project_path, 'res/io_paddle.json')
    quantize_config = os.path.join(project_path, 'res/test_paddle_detection1.json')
    custum_map = 'custum_kernel_map1.json'
    update_io_config(io_config, shape)

    # only for model split
    if split:
        io_config = os.path.join(model_dir, 'output/io_paddle.json')
        custum_map = 'custum_kernel_map2.json'
        quantize_config = os.path.join(project_path, 'res/test_paddle_detection2.json')
    update_quantize_config(quantize_config, model_dir)

    print('****************************************')
    print('************** reader **************')
    print('****************************************')
    ret = subprocess.run(
        ['python3',
        'reader.py',
        model_dir,
        shape,
        os.path.join(model_dir, 'image')
    ])
    assert(ret.returncode == 0)

    if split:
        print('****************************************')
        print('**************split**************')
        print('****************************************')
        ret = subprocess.run([
            'python3',
            'split.py',
            '--model_dir',
            os.path.join(model_dir, 'model'),
            '--input_shape_dict',
            shape
        ])
        assert(ret.returncode == 0)
        
    print('****************************************')
    print('**************build**************')
    print('****************************************')

    ret = subprocess.run([
        '/opt/new_secpy/x86_64/pyenc',
        '/opt/compiler/compile.pyxes',
        config_file
    ])
    print('****************************************')
    print('**************pack **************')
    print('****************************************')
    package(model_dir=model_dir, split=split)                                

