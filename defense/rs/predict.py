""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse
from core import Smooth
import torch
from architectures import get_architecture
from datasets import load_images, get_labels, load_labels
import tqdm
import os

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("input", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("label_file", type=str, help="path to the label file")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1, help="batch size")
parser.add_argument("--skip", type=int, default=400, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=400, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--GPU_ID',default='0',type=str, help='GPU_ID')
parser.add_argument("--targeted", action='store_true', help='targeted attack')
args = parser.parse_args()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], 'imagenet')
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    f2l = load_labels(args.label_file, args.targeted)
    smoothed_classifier = Smooth(base_classifier, 1000, args.sigma)

    # prepare output file
    succ = total = 0.
    all_predictions = []
    all_filenames = []

    for batch_idx, [filenames, images] in tqdm.tqdm(enumerate(load_images(args.input, args.batch))):
        labels = get_labels(filenames, f2l)
        images = images.cuda()

        # 处理batch中的每个图像
        for i in range(images.shape[0]):
            img = images[i:i+1]  # 保持维度
            # 获取label并转换为标量
            if isinstance(labels, torch.Tensor):
                label = labels[i].item()
            else:
                label = labels[i]
            filename = filenames[i] if isinstance(filenames, list) else filenames

            # predict返回0-999的类别索引，或者ABSTAIN(-1)
            prediction = smoothed_classifier.predict(img, args.N, args.alpha, 1)

            # 保存预测结果（转换为1-1000以匹配HGD格式）
            if prediction == Smooth.ABSTAIN:
                pred_output = -1  # ABSTAIN标记为-1，便于识别
            else:
                pred_output = int(prediction) + 1  # 转换为1-1000

            all_predictions.append(pred_output)
            all_filenames.append(os.path.basename(filename))

            # 统计成功率（prediction和label都是0-999）
            # ABSTAIN算作分类错误
            if prediction != Smooth.ABSTAIN and prediction == label:
                succ += 1
            total += 1

        if total % 200 == 0 and total > 0:
            if not args.targeted:
                print("Attack Success Rate: {:.2f}%".format(100. * (total - succ) / total))
            else:
                print("Attack Success Rate: {:.2f}%".format(100. * succ / total))

    # 写入输出文件
    with open(args.outfile, 'w') as f:
        for filename, pred in zip(all_filenames, all_predictions):
            f.write(f'{filename},{pred}\n')

    print(args.input)
    if not args.targeted:
        print("=>Final Attack Success Rate: {:.2f}%".format(100. * (total - succ) / total))
    else:
        print("=>Final Attack Success Rate: {:.2f}%".format(100. * succ / total))