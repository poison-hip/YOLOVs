# coding=utf-8
import paddle.fluid as fluid
import paddle
import os
from paddle.fluid.layer_helper import LayerHelper
import argparse
import paddle2onnx
import subprocess
import json

paddle.enable_static()
place = fluid.CPUPlace()
exe = fluid.Executor(place)


def new_prepend_feed_ops(inference_program, feed_target_names, feed_holder_name="feed"):
    import paddle.fluid.core as core

    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True,
    )

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            print(
                "The input[{i}]: '{name}' doesn't exist in pruned inference"
                "program, which will be ignored in new saved model.".format(
                    i=i, name=name
                )
            )
            continue
        out = global_block.var(name)
        global_block._prepend_op(
            type="feed",
            inputs={"X": [feed_var]},
            outputs={"Out": [out]},
            attrs={"col": i},
        )


def construct_post(model_path, yolo_shape_dict, class_num):
    anchors = [
        [116, 90, 156, 198, 373, 326],
        [30, 61, 62, 45, 59, 119],
        [10, 13, 16, 30, 33, 23],
    ]
    ratios = [32, 16, 8]
    program = fluid.Program()
    with fluid.program_guard(main_program=program):
        x = [
            paddle.static.data(name=i, shape=yolo_shape_dict[i], dtype="float32")
            for i in yolo_shape_dict.keys()
        ]
        im_shape = paddle.static.data(name="im_shape", shape=[1, 2], dtype="float32")
        scale_factor = paddle.static.data(
            name="scale_factor", shape=[1, 2], dtype="float32"
        )
        shape = paddle.cast(im_shape / scale_factor, "int32")
        t = [
            paddle.vision.ops.yolo_box(
                x[i], shape, anchors[i], class_num, 0.004999999888241291, ratios[i]
            )
            for i in range(3)
        ]
        score = [paddle.transpose(t[i][1], [0, 2, 1]) for i in range(3)]
        score = paddle.concat(score, 2)
        bbox = paddle.concat([t[i][0] for i in range(3)], 1)

    fluid.io.save_inference_model(
        model_path,
        [*yolo_shape_dict.keys(), "im_shape", "scale_factor"],
        [bbox, score],
        exe,
        main_program=program,
    )
    os.rename(
        os.path.join(model_path, "__model__"), os.path.join(model_path, "post.pdmodel")
    )
    return os.path.join(model_path, "post.pdmodel")


def get_split_names(prog):
    yolo_names = []
    nms_names = []
    for block in prog.blocks:
        for op in block.ops:
            if op.type == "yolo_box":
                yolo_names.append(op.input("X")[0])
            if op.type == "multiclass_nms3":
                nms_names.append(op.input("BBoxes")[0])
                nms_names.append(op.input("Scores")[0])
    return yolo_names, nms_names


def append_fetch_ops(program, fetch_target_names, fetch_holder_name="fetch"):
    import paddle.fluid.core as core

    global_block = program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name, type=core.VarDesc.VarType.FETCH_LIST, persistable=True
    )
    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type="fetch",
            inputs={"X": [name]},
            outputs={"Out": [fetch_var]},
            attrs={"col": i},
        )


def insert_fetch(program, fetchs, fetch_holder_name="fetch"):
    global_block = program.global_block()
    need_to_remove_op_index = list()
    for i, op in enumerate(global_block.ops):
        if op.type == "fetch":
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()


def process_old_ops_desc(program):
    for i in range(len(program.blocks[0].ops)):
        if program.blocks[0].ops[i].type == "matmul":
            if not program.blocks[0].ops[i].has_attr("head_number"):
                program.blocks[0].ops[i]._set_attr("head_number", 1)


def infer_shape(program, input_shape_dict):
    OP_WITHOUT_KERNEL_SET = {
        "feed",
        "fetch",
        "recurrent",
        "go",
        "rnn_memory_helper_grad",
        "conditional_block",
        "while",
        "send",
        "recv",
        "listen_and_serv",
        "fl_listen_and_serv",
        "ncclInit",
        "select",
        "checkpoint_notify",
        "gen_bkcl_id",
        "c_gen_bkcl_id",
        "gen_nccl_id",
        "c_gen_nccl_id",
        "c_comm_init",
        "c_sync_calc_stream",
        "c_sync_comm_stream",
        "queue_generator",
        "dequeue",
        "enqueue",
        "heter_listen_and_serv",
        "c_wait_comm",
        "c_wait_compute",
        "c_gen_hccl_id",
        "c_comm_init_hccl",
        "copy_cross_scope",
    }
    for k, v in input_shape_dict.items():
        program.blocks[0].var(k).desc.set_shape(v)
    for i in range(len(program.blocks)):
        for j in range(len(program.blocks[0].ops)):
            if program.blocks[i].ops[j].type in OP_WITHOUT_KERNEL_SET:
                continue
            program.blocks[i].ops[j].desc.infer_shape(program.blocks[i].desc)


def get_new_shape(prog, input_shape_dict, names):
    names = (*names[0], *names[1])
    process_old_ops_desc(prog)
    infer_shape(prog, input_shape_dict)
    get_split_names(prog)
    block = prog.blocks[0]
    shape_dict = {name: block.vars[name].shape for name in names}
    dtype_dict = {name: block.vars[name].dtype.name for name in names}
    return shape_dict


def construct_nms(model_path, shape, nms_names, class_num):
    def multiclass_nms3(
        bboxes,
        scores,
        score_threshold,
        nms_top_k,
        keep_top_k,
        nms_threshold=0.3,
        normalized=True,
        nms_eta=1.0,
        background_label=0,
        name=None,
    ):
        helper = LayerHelper("multiclass_nms3", **locals())
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype="int32")
        nms_rois_num = helper.create_variable_for_type_inference(dtype="int32")

        inputs = {"BBoxes": bboxes, "Scores": scores}
        outputs = {"Out": output, "Index": index, "NmsRoisNum": nms_rois_num}
        helper.append_op(
            type="multiclass_nms3",
            inputs=inputs,
            attrs={
                "background_label": background_label,
                "score_threshold": score_threshold,
                "nms_top_k": nms_top_k,
                "nms_threshold": nms_threshold,
                "keep_top_k": keep_top_k,
                "nms_eta": nms_eta,
                "normalized": normalized,
            },
            outputs=outputs,
        )
        return output, index, nms_rois_num

    background_label = class_num
    keep_top_k = 100
    nms_eta = 1
    nms_threshold = 0.44999998807907104
    nms_top_k = 1000
    normalized = True
    score_threshold = 0.009999999776482582
    program = fluid.Program()
    with fluid.program_guard(main_program=program):
        box = paddle.static.data(name="bboxes", shape=shape[0], dtype="float32")
        score = paddle.static.data(name="scores", shape=shape[1], dtype="float32")
        output, _, _ = multiclass_nms3(
            box,
            score,
            score_threshold=score_threshold,
            nms_top_k=nms_top_k,
            keep_top_k=keep_top_k,
            nms_eta=nms_eta,
            background_label=background_label,
            normalized=normalized,
            nms_threshold=nms_threshold,
        )
    fluid.io.save_inference_model(
        model_path,
        ["bboxes", "scores"],
        [output],
        exe,
        main_program=program,
    )
    os.rename(
        os.path.join(model_path, "__model__"), os.path.join(model_path, "nms.pdmodel")
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Path of directory saved the input model."
    )
    parser.add_argument(
        "--input_shape_dict", required=True, help="The input model input shape dict."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_shape_dict_str = args.input_shape_dict
    input_shape_dict = eval(input_shape_dict_str)
    input_shape_dict = {
        "image": [1, 3, *input_shape_dict],
        "scale_factor": [1, 2],
        "im_shape": [1, 2],
    }

    prog, _, _ = fluid.io.load_inference_model(
        args.model_dir,
        exe,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams",
    )

    names = get_split_names(prog)
    output_dict = get_new_shape(prog, input_shape_dict, names)

    # get class_num
    class_num = 0
    for i in output_dict:
        if output_dict[i][2] == 6300:
            class_num = output_dict[i][1]
            break
    assert class_num != 0, "invalid class number"

    subprocess.run(
        [
            "python3",
            "prune_paddle_model.py",
            "--model_dir",
            args.model_dir,
            "--model_filename",
            "model.pdmodel",
            "--params_filename",
            "model.pdiparams",
            "--output_names",
            *names[0],
            "--save_dir",
            os.path.join(args.model_dir, "../output"),
        ]
    )
    model_path = os.path.join(args.model_dir, "../output")
    onnx_model = construct_post(
        model_path, {i: output_dict[i] for i in names[0]}, class_num
    )
    construct_nms(model_path, [output_dict[i] for i in names[1]], names[1], class_num)
    paddle2onnx.export(onnx_model, save_file=os.path.join(model_path, "post.onnx"))
    os.remove(os.path.join(model_path, "post.pdmodel"))
    print("All models have been exported at %s" % model_path)

    with open(os.path.join(model_path, "io_paddle.json"), "w") as f:
        config = []
        config.append(
            {
                "type": "INPUT",
                "name": "image",
                "shape": input_shape_dict["image"],
                "layout": "NCHW",
                "dtype": "float32",
            }
        )
        for i in names[0]:
            config.append(
                {
                    "type": "OUTPUT",
                    "name": i,
                    "shape": output_dict[i],
                    "dtype": "float32",
                }
            )

        config.extend(
            [
                {"type": "nms_in", "name": "bboxes", "shape": output_dict[names[1][0]]},
                {"type": "nms_in", "name": "scores", "shape": output_dict[names[1][1]]},
            ]
        )

        # TODO read from onnx model
        config.extend(
            [
                {
                    "type": "post_out",
                    "name": "concat_1.tmp_0",
                    "shape": output_dict[names[1][0]],
                },
                {
                    "type": "post_out",
                    "name": "concat_0.tmp_0",
                    "shape": output_dict[names[1][1]],
                },
            ]
        )

        json.dump(config, f, indent=4)
