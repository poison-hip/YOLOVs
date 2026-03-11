import tvm
from tvm import relay
import paddle
import paddle.fluid as fluid
import paddle.static as static
import json
import sys, os

def get_feeds(config):
    feed_dict = {}
    with open(config, 'r') as f:
        conf = json.load(f)
        for i in conf:
            if i['type'] == 'nms_in':
                feed_dict[i['name']] = i['shape']
    return feed_dict

def build(model_path, feed_dict):
    exe = static.Executor(paddle.CPUPlace())
    paddle.enable_static()
    [paddle_prog, feed, fetch] = fluid.io.load_inference_model(model_path, exe, 'nms.pdmodel')
    mod, params = tvm.relay.frontend.from_paddle(paddle_prog, shape_dict = feed_dict)

    with tvm.transform.PassContext(opt_level=3):
        target = 'llvm -mtriple=aarch64-linux-gnu'
        lib = relay.build(mod, target, params = params)
        out = model_path + '/nms.tar'
        lib.export_library(out)

if __name__ == "__main__":
    model_dir = sys.argv[1]
    # nms_model_dir = os.path.join(model_dir, 'output')
    config_file = os.path.join(model_dir, 'io_paddle.json')
    feed_dict = get_feeds(config_file)
    build(model_dir, feed_dict)

    
