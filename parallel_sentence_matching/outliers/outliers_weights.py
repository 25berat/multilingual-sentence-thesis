from collections import defaultdict
import torch
from transformers import AutoModel
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Code used to:
# a) Identify magnitude-wise outliers for XLM-R
# b) Identify magnitude-wise outliers per each language (used in chapter 5) + generate visualizations

parser = argparse.ArgumentParser(description='Establish outlier dimensions.')
parser.add_argument('--model', type=str, help="name of the model")
parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
parser.add_argument('--stdevs', type=int, default=3,
                    help="dimension must be this number of standard deviations from mean to meet outlier definition")
parser.add_argument('--type', type=str, help="to analyze per-model or per-language ('all' or 'language')")
args = parser.parse_args()

model = AutoModel.from_pretrained(args.model)

if args.type == "all":
    layer = model.encoder.layer[args.layer]
    bias = layer.output.LayerNorm.bias.data
    weight = layer.output.LayerNorm.weight.data

    mean_bias = torch.mean(bias)
    std_dev_bias = torch.std(bias)

    mean_weight = torch.mean(weight)
    std_dev_weight = torch.std(weight)

    counter_bias = 0
    outliers_bias = []

    for i in bias:
        if abs(i) > mean_bias + args.stdevs * std_dev_bias:
            outliers_bias.append(counter_bias)
        counter_bias += 1

    print(f"Outlier dimensions: {outliers_bias}")
    for o in outliers_bias:
        print(f"Average value of {o}: {bias[o]}")

    counter = 0
    outliers = []

    for i in bias:
        if abs(i) > mean_weight + args.stdevs * std_dev_weight:
            outliers.append(counter)
        counter += 1

    print(f"Outlier dimensions: {outliers}")
    for o in outliers:
        print(f"Average value of {o}: {weight[o]}")

    layer

