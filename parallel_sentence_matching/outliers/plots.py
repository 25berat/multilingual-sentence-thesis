import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from constants import langs_tatoeba, langs_wiki, lang_dict_3_2, lang_dict


# produce heatmaps of sentence embeddings per language (not featured in thesis)
def heatmaps(args, tgt_embs, eng_embs, tgt_lang):
    fig_dir = f'../plots/{args.dataset}/{args.model}/{args.layer}/{tgt_lang}/'
    os.makedirs(os.path.dirname(fig_dir), exist_ok=True)

    fig = plt.figure(figsize=(8, 2))
    ax = sns.heatmap(tgt_embs, cmap="YlGnBu", yticklabels=False, xticklabels=100)
    plt.gca().invert_yaxis()
    plt.title(f"{tgt_lang} representations")
    plt.ylabel("# sentence")
    plt.xlabel("dimension")
    plt.savefig(f'{fig_dir}/heatmap_{tgt_lang}{args.append_file_name}.png', dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close(fig)

    if eng_embs is not None:
        fig = plt.figure(figsize=(8, 2))
        ex = sns.heatmap(eng_embs, cmap="YlGnBu", yticklabels=False, xticklabels=100)
        plt.gca().invert_yaxis()
        plt.title("eng representations")
        plt.ylabel("# sentence")
        plt.xlabel("dimension")
        plt.savefig(f'{fig_dir}/heatmap_eng{args.append_file_name}.png', dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close(fig)
    print(f"Finished {tgt_lang} heatmaps.")


# produce visualizations of language level mean embeddings (for target and english)
def mean_vec(args, tgt_embs: torch.Tensor, eng_embs: torch.Tensor, tgt_lang, hidden_size=768):
    fig_dir = f'../plots/{args.dataset}/{args.model}/{args.layer}/{tgt_lang}/'
    os.makedirs(os.path.dirname(fig_dir), exist_ok=True)

    tgt_mean_vec = torch.mean(tgt_embs, axis=0)

    clrs = sns.color_palette("pastel", 8)
    mean = torch.mean(tgt_mean_vec)
    std_dev = torch.std(tgt_mean_vec)
    x = np.arange(hidden_size)

    fig = plt.figure(figsize=(4, 2))
    plt.xlim([0, hidden_size])
    plt.plot(tgt_mean_vec)
    plt.fill_between(x, mean - std_dev * args.stdevs, mean + std_dev * args.stdevs, alpha=0.5, facecolor=clrs[7])
    plt.title(f"{tgt_lang} mean representation")
    plt.xlabel("dimension")
    plt.savefig(f'{fig_dir}/mean_{tgt_lang}{args.append_file_name}.png', dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close(fig)

    if eng_embs is not None:
        eng_mean_vec = torch.mean(eng_embs, axis=0)
        mean = torch.mean(eng_mean_vec)
        std_dev = torch.std(eng_mean_vec)

        x = np.arange(hidden_size)
        fig = plt.figure(figsize=(4, 2))
        plt.xlim([0, hidden_size])
        plt.plot(eng_mean_vec)
        plt.fill_between(x, mean - std_dev * args.stdevs, mean + std_dev * args.stdevs, alpha=0.5, facecolor=clrs[7])
        plt.title("eng mean representation")
        plt.xlabel("dimension")
        plt.savefig(f'{fig_dir}/mean_eng{args.append_file_name}.png', dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close(fig)
    print(f"Finished {tgt_lang} mean vecs.")


# Analyze average english embedding
# ref fig 5.4 in thesis
def mean_english(args, langs, hidden_size=768):
    if not args.dataset == 'tatoeba':
        raise ValueError("cannot compute parallel english for non-parallel data")
    fig_dir = f'../plots/{args.dataset}/{args.model}/{args.layer}/'
    os.makedirs(os.path.dirname(fig_dir), exist_ok=True)
    mean_sum_vec = None
    for lang in langs:
        print(f"Processing parallel English of: {lang}")
        eng_emb = torch.load(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng{args.append_file_name}.pt')
        if mean_sum_vec is None:
            mean_sum_vec = torch.mean(eng_emb, axis=0)
        else:
            mean_sum_vec = torch.add(mean_sum_vec, torch.mean(eng_emb, axis=0))

    mean_vector = torch.divide(mean_sum_vec, len(langs))
    mean = torch.mean(mean_vector)
    std_dev = torch.std(mean_vector)
    x = np.arange(hidden_size)
    clrs = sns.color_palette("pastel", 8)

    counter = 0
    outliers = []
    for i in mean_vector:
        if abs(i) > mean + 3 * std_dev:
            outliers.append(counter)
        counter += 1

    print(f"Outlier dimensions: {outliers}")
    for o in outliers:
        print(f"Average value of {o}: {mean_vector[o]}")

    plt.figure(figsize=(6, 1.5))
    plt.xlim([0, hidden_size])
    plt.plot(mean_vector)
    plt.fill_between(x, mean - std_dev * args.stdevs, mean + std_dev * args.stdevs, alpha=0.5, facecolor=clrs[7])
    plt.title(f"average english embedding")
    plt.xlabel("dim")
    plt.savefig(f'{fig_dir}/avg_eng_embedding{args.append_file_name}.png', dpi=300, bbox_inches="tight")


def mean_all(args, tgt_embs, hidden_size=768):
    all_mean = torch.stack([emb.mean(axis=0) for emb in tgt_embs]).mean(axis=0)

    fig_dir = f'../plots/{args.dataset}/{args.model}/{args.layer}/'
    os.makedirs(os.path.dirname(fig_dir), exist_ok=True)

    mean = torch.mean(all_mean)
    std_dev = torch.std(all_mean)
    x = np.arange(hidden_size)
    clrs = sns.color_palette("pastel", 8)

    plt.figure(figsize=(6, 1.5))
    plt.xlim([0, hidden_size])
    plt.plot(all_mean)
    plt.fill_between(x, mean - std_dev * args.stdevs, mean + std_dev * args.stdevs, alpha=0.5, facecolor=clrs[7])
    plt.title(f"Layer {args.layer} mean embedding")
    plt.xlabel("dim")
    plt.savefig(f'{fig_dir}/all_mean_embedding{args.append_file_name}.png', dpi=300, bbox_inches="tight")


# plot sentence level outliers (single values in an outlier dimension)
# can plot either only target lang values or together with english aligned ones
def plot_outliers(args, tgt_embs, eng_embs, tgt_lang):
    dir_tgt = f'../plots/{args.dataset}/{args.model}/{args.layer}/{tgt_lang}/{tgt_lang}_dim_{args.dim}.png'
    dir_tgt_en = f'../plots/{args.dataset}/{args.model}/{args.layer}/{tgt_lang}/eng_{tgt_lang}_dim_{args.dim}.png'
    os.makedirs(os.path.dirname(dir_tgt), exist_ok=True)
    os.makedirs(os.path.dirname(dir_tgt_en), exist_ok=True)

    """dim_values = tgt_embs.T[args.dim][:-1]
    print(f"Max: {torch.amax(dim_values)}")
    print(f"Min: {torch.amin(dim_values)}")"""

    plt.figure(figsize=(5, 1.5))
    plt.plot(tgt_embs.T[args.dim][:-1],
             marker='x', linestyle='None', markersize=2, color='xkcd:soft green', label=tgt_lang)
    plt.title(f"{lang_dict_3_2[tgt_lang]} dimension {args.dim}")
    if eng_embs is not None:
        plt.plot(eng_embs.T[args.dim][:-1], marker='x', linestyle='None', markersize=2, color='xkcd:azure', label='en')
        plt.title(f"{lang_dict_3_2[tgt_lang]}-en dimension {args.dim}")
    plt.xlabel("#sentence")
    plt.savefig(dir_tgt_en, dpi=300, bbox_inches="tight")
    plt.close()


def main(args):
    if args.dataset == "tatoeba":
        langs = langs_tatoeba
    elif args.dataset == "wiki":
        langs = langs_wiki
    elif args.dataset == "sts":
        return
    else:
        raise ValueError("unknown dataset argument")

    if args.job == "heatmaps":
        print("Generating heatmaps.")
        for lang in langs:
            # print(f"Current language: {lang}")
            target_emb = torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{args.append_file_name}.pt')
            eng_emb = None if args.dataset == "wiki" else torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng{args.append_file_name}.pt')
            heatmaps(args, target_emb, eng_emb, lang)

    elif args.job == "means":
        print("Generating mean representations.")
        tgt_embs = []
        eng_embs = []
        for lang in langs:
            # print(f"Current language: {lang}")
            target_emb = torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{args.append_file_name}.pt')
            eng_emb = None if args.dataset == "wiki" else torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng{args.append_file_name}.pt')
            tgt_embs.append(target_emb)
            eng_embs.append(eng_emb)
            mean_vec(args, target_emb, eng_emb, lang)
        if eng_embs[0] is not None:
            tgt_embs.extend(eng_embs)
        mean_all(args, tgt_embs)

    elif args.job == "outlier":
        print("Generating outlier visualization.")
        if args.language != "":
            lang = lang_dict[args.language]
            target_emb = torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{args.append_file_name}.pt')
            eng_emb = None if args.dataset == "wiki" else torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng{args.append_file_name}.pt')
            plot_outliers(args, target_emb, eng_emb, lang)
        else:
            for lang in langs:
                # print(f"Current language: {lang}")
                target_emb = torch.load(
                    f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{args.append_file_name}.pt')
                eng_emb = None if args.dataset == "wiki" else torch.load(
                    f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng{args.append_file_name}.pt')
                plot_outliers(args, target_emb, eng_emb, lang)

    elif args.job == "mean_english":
        mean_english(args, langs)
    else:
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embedding visualizations.')
    parser.add_argument('--model', type=str, help="name of the model (xlm-r, x2s-cca or x2s-mse)")
    parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki"],
                        help="use embeddings from this dataset (tatoeba, wiki)")
    parser.add_argument('--stdevs', type=int, default=3,
                        help="dimension must be this number of standard deviations from mean to meet outlier definition")
    parser.add_argument('--dim', type=int, nargs='?', default=588, help='dimension to analyze, e.g. 588 (for "outlier")')
    parser.add_argument('--append_file_name', type=str, default="", help='to load files à la .._whitened.pt')
    parser.add_argument('--job', type=str, choices=['heatmaps', 'means', 'outlier', 'mean_english'],
                        help="'heatmaps' or 'means' or 'outlier' or 'mean_english'")
    parser.add_argument('--language', type=str, nargs='?', default="",
                        help="if only a specific language should be considered")
    args = parser.parse_args()
    main(args)
