#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.datasets import ImageFolder
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS, RECTANGULAR_TRANSFORMS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def detect_directory(model_path, weights_path, img_path, classes, output_path,
                     batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5,
                     max_det=300, nms_type="nms"):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    # 函数作用：对目录下所有图片执行检测，并将可视化结果保存到输出目录。
    dataloader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    img_detections, imgs = detect(
        model,
        dataloader,
        output_path,
        conf_thres,
        nms_thres,
        max_det=max_det,
        nms_type=nms_type)
    _draw_and_save_output_images(
        img_detections, imgs, img_size, output_path, classes)

    print(f"---- Detections were saved to: '{output_path}' ----")


def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5, max_det=300, nms_type="nms"):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    # 函数作用：对单张图片做前向推理并返回检测框坐标。
    model.eval()  # Set model to evaluation mode

    # Configure input
    # 与训练时保持一致：非方形输入使用矩形预处理路径。
    base_transform = RECTANGULAR_TRANSFORMS if isinstance(img_size, (tuple, list)) and img_size[0] != img_size[1] else DEFAULT_TRANSFORMS
    input_img = transforms.Compose([
        base_transform,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)

    if torch.cuda.is_available():
        input_img = input_img.to("cuda")

    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        # NMS 去除重复框，并保留置信度更高的候选框。
        detections = non_max_suppression(detections, conf_thres, nms_thres, max_det=max_det, nms_type=nms_type)
        # 将检测框从模型输入坐标映射回原图坐标。
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()


def detect(model, dataloader, output_path, conf_thres, nms_thres, max_det=300, nms_type="nms"):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    # 函数作用：按 batch 对输入图片做推理，返回每张图的检测结果及路径。
    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    # 统一 CPU/CUDA 张量类型，避免在循环中频繁分支判断。
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres, max_det=max_det, nms_type=nms_type)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs


def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    """Draws detections in output images and stores them.

    :param img_detections: List of detections
    :type img_detections: [Tensor]
    :param imgs: List of paths to image files
    :type imgs: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """

    # 函数作用：遍历检测结果，逐张图调用绘制函数并保存。
    # Iterate through images and save plot of detections
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        _draw_and_save_output_image(
            image_path, detections, img_size, output_path, classes)


def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # 函数作用：将单张图的检测框和类别文字画到图上并保存为 png。
    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    # 同一张图里同一类别使用固定颜色，便于快速查看。
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=f"{classes[int(cls_pred)]}: {conf:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    # 函数作用：构建推理阶段 DataLoader，保证批量与单图预处理一致。
    # Mirror detect_image preprocessing so batch and single-image inference stay consistent.
    base_transform = RECTANGULAR_TRANSFORMS if isinstance(img_size, (tuple, list)) and img_size[0] != img_size[1] else DEFAULT_TRANSFORMS
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([base_transform, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def run():
    # 函数作用：推理脚本命令行入口，解析参数后触发目录批量检测。
    print_environment_info()
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-i", "--images", type=str, default="data/samples", help="Path to directory with images to inference")
    parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Square image size for yolo")
    parser.add_argument("--img_width", type=int, default=None, help="Optional non-square image width")
    parser.add_argument("--img_height", type=int, default=None, help="Optional non-square image height")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum detections per image after NMS")
    parser.add_argument("--nms_type", type=str, default="nms", choices=["nms", "diou"], help="NMS type")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    # 同时传入宽和高时，启用非方形输入推理。
    if args.img_width is not None and args.img_height is not None:
        img_size = (args.img_height, args.img_width)
    else:
        img_size = args.img_size

    detect_directory(
        args.model,
        args.weights,
        args.images,
        classes,
        args.output,
        batch_size=args.batch_size,
        img_size=img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        max_det=args.max_det,
        nms_type=args.nms_type)


if __name__ == '__main__':
    run()
