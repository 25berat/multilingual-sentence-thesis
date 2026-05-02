""" Adapted from https://github.com/Sara-Rajaee/Multilingual-Isotropy
We did not use this in our final experiments but rather refer to this and refactored it slightly to understand and
reproduce the paper it came from.
"""

import argparse
import math as mt
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf
import torch
from matplotlib.pyplot import figure
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoConfig

from post_processing import cluster_based


def text2rep(model, tokenizer, df):
    representation = []
    with torch.no_grad():
        for i in range(len(df)):
            inputs = tokenizer(df['Sentence'].iloc[i], add_special_tokens=False, return_tensors='pt')
            # not sure what the below does
            # inputs = np.asarray(inputs, dtype='int32').reshape((1, -1))
            # print(inputs)
            output = model(**inputs)[0]
            for j in range(output.shape[1]):
                representation.append(output[0][j])

    return representation


def isotropy(representations):
    """Calculating isotropy of embedding space based on I_PC
           arg:
              representations (n_samples, n_dimensions)
            """

    eig_values, eig_vectors = np.linalg.eig(np.matmul(np.transpose(representations), representations))
    max_f = -mt.inf
    min_f = mt.inf

    for i in range(eig_vectors.shape[1]):
        f = np.matmul(representations, np.expand_dims(eig_vectors[:, i], 1))
        f = np.sum(np.exp(f))

        min_f = min(min_f, f)
        max_f = max(max_f, f)

    isotropy = min_f / max_f

    return isotropy


def cosine_contribution(representation, n):
    contribution = []
    num = 1000
    for i in range(num):
        n1 = random.randint(0, len(representation) - 1)
        n2 = random.randint(0, len(representation) - 1)
        if n1 != n2:
            contribution.append(np.multiply(representation[n1], representation[n2]) /
                                (np.linalg.norm(representation[n1]) * np.linalg.norm(representation[n2])))

    sim = []
    for i in range(num):
        n1 = random.randint(0, len(representation) - 1)
        n2 = random.randint(0, len(representation) - 1)
        if n1 != n2:
            sim.append(cosine_similarity(
                np.reshape(representation[n1], (1, -1)), np.reshape(representation[n2], (1, -1)))[0][0])

    ary = np.mean(contribution, axis=0) / np.mean(sim)
    return ary[np.argsort(ary)[-n:]]


def extract_reps_per_lang(model, tokenizer, df, lang, args):
    file_name = f'{args.base_dir}/{args.model_name}/cbie-{args.cbie}_selected_{lang}.npy'
    if os.path.isfile(file_name) and not args.overwrite:
        return

    print(f"extracting {args.model} representations for {lang}")
    rep = text2rep(model, tokenizer, df)
    rd = random.sample(range(0, len(rep)), 10000)
    selected = np.stack([np.asarray(rep[idx]) for idx in rd])  # todo isn't that the same as = rep[rd]?

    if args.do_cbie:
        selected = cluster_based(selected, args.n_cluster, args.n_pc, args.hidden_size)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    np.save(file_name, selected)


def get_outliers(selected, stdevs):
    mean_vec = torch.mean(selected, axis=0)
    mean = torch.mean(mean_vec)
    std_dev = torch.std(mean_vec)
    outliers = []
    for counter, i in enumerate(mean_vec):
        if abs(i) > mean + stdevs * std_dev:
            outliers.append(counter)


def get_contributions(langs: List[str], base_dir, model_name):
    model_contribution = []
    for lang in langs:
        selected = np.load(f'{base_dir}/{model_name}/selected_{lang}.npy', allow_pickle=True)
        model_contribution.append(cosine_contribution(selected, 3))
        print(f'{lang} isotropy score: {isotropy(selected)}')
        print(f'{lang} outlier dims: {get_outliers(selected)}')
    model_contribution = np.asarray(model_contribution)

    t_file = open(f'{model_name}-contribution.txt', "w")
    for row in model_contribution:
        np.savetxt(t_file, row)
    t_file.close()


def vis_outliers(lang, base_dir, model_name, fig_dir, hidden_size=768):
    selected = np.load(f'{base_dir}/{model_name}/selected_{lang}.npy', allow_pickle=True)
    os.makedirs(os.path.dirname(f'{fig_dir}/{model_name}/'), exist_ok=True)
    plt.rcParams["figure.figsize"] = (30, 3)
    x = np.arange(hidden_size)
    print(type(selected))
    st = np.std(selected) * 3
    m = np.mean(selected)

    figure(figsize=(23, 3), dpi=80)
    fig = plt.bar(x, np.mean(selected, axis=0), color='#073b4c', width=2)
    plt.xlim([0, hidden_size])
    plt.ylim([-2, 2])
    plt.xticks(fontsize=30)
    clrs = sns.color_palette("pastel", 8)
    plt.yticks(fontsize=30)
    plt.yticks(np.arange(-2, 2.1, 2))
    plt.fill_between(x, m - st, m + st, alpha=0.5, facecolor=clrs[7])

    plt.title(f"{lang}")
    plt.savefig(f'{fig_dir}/{model_name}/{lang}.png', dpi=300, bbox_inches="tight")
    plt.show()


# Loading mBERT
# i use pytorch and automodel instead of TF, would that somehow change the results??
# casing = "bert-base-multilingual-uncased"
# tokenizer_mBERT = BertTokenizer.from_pretrained(casing, do_lower_case=True, add_special_tokens=True)
# config = BertConfig(casing, output_hidden_states=True)
# mBERT = TFBertModel.from_pretrained(casing)
# mBERT.trainable = False

def load_model(model_name="bert-base-multilingual-uncased"):
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True,
                                              add_special_tokens=True)  # do_lower_case=False for other models
    model = AutoModel.from_pretrained(model_name)  # TFAutoModel.from_pretrained(model_name)
    model.trainable = False
    return config, tokenizer, model


def main():
    parser = argparse.ArgumentParser(description='Isotropy analysis, ref. Rajaee and Pilehvar 2022')
    parser.add_argument('--model', type=str, default="bert-base-multilingual-uncased", help="model name")
    parser.add_argument('--do_lowercase', type=bool, default=True, help="whether to lowercase during tokenising")
    parser.add_argument('--hidden_size', type=int, default=768, help="model hidden size")
    parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
    parser.add_argument('--base_dir', type=str, help="where to save/load representations for the analysis")
    parser.add_argument('--fig_dir', type=str, help="where to save figures")
    parser.add_argument('--do_cbie', action='store_true', help="if True, apply cbie before analysis")
    # defaults taken from R&P's paper
    parser.add_argument('--n_cluster', type=int, default=27, help="if cbie, number of clusters to create")
    parser.add_argument('--n_pc', type=int, default=12, help="if cbie, number of principal components to discard")
    parser.add_argument('--overwrite', type=bool, default=False,
                        help="whether to redo extracting embeddings for which a file already exists")
    args = parser.parse_args()

    _, tokenizer, model = load_model(args.model)

    # Loading Wikipedia datasets
    df_su = pd.read_csv('../data/Wikipedia/Sundanese.csv', sep=',')
    df_sw = pd.read_csv('../data/Wikipedia/Swahili.csv', sep=',')
    df_en = pd.read_csv('../data/Wikipedia/English.csv', sep=',')
    df_es = pd.read_csv('../data/Wikipedia/Spanish.csv', sep=',')
    df_ar = pd.read_csv('../data/Wikipedia/Arabic.csv', sep=',')
    df_tr = pd.read_csv('../data/Wikipedia/Turkish.csv', sep=',')

    langs = ['su', 'sw', 'en', 'es', 'ar', 'tr']
    dfs = [df_su, df_sw, df_en, df_es, df_ar, df_tr]

    for lang, df in zip(langs, dfs):
        extract_reps_per_lang(model, tokenizer, df, lang, args)
        vis_outliers(lang, args.base_dir, args.model, args.fig_dir, args.hidden_size)
    get_contributions(langs, args.base_dir, args.model)


if __name__ == "__main__":
    main()
