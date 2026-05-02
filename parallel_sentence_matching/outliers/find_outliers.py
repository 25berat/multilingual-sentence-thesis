import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

# Code used to:
# a) Identify magnitude-wise outliers for XLM-R
# b) Identify magnitude-wise outliers per each language (used in chapter 5) + generate visualizations
from constants import langs_tatoeba, langs_wiki, lang_dict_3_2, sts_tracks


def main(args):
    if args.dataset == "tatoeba":
        langs = langs_tatoeba
    elif args.dataset == "wiki":
        langs = langs_wiki
    elif args.dataset == "sts":
        langs = []
        for track in sts_tracks:
            langs.append((track, "lng1"))
            langs.append((track, "lng2"))
    else:
        raise ValueError("unknown dataset argument")

    # ANALYZE ONE AVG EMBEDDING OVER ALL LANGUAGES
    if args.type == "all":
        mean_sum_vec = None
        for lang in langs:
            # print(f"Considering language {lang}.")
            if type(lang) is tuple:
                lang_or_track = lang[0]
                lang = lang[1]
            else:
                lang_or_track = lang

            target_emb = torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang_or_track}/{lang}{args.append_file_name}.pt')
            if mean_sum_vec is None:
                mean_sum_vec = torch.mean(target_emb, axis=0)
            else:
                mean_sum_vec = torch.add(mean_sum_vec, torch.mean(target_emb, axis=0))

        mean_vec_all_langs = torch.divide(mean_sum_vec, len(langs))
        mean = torch.mean(mean_vec_all_langs)
        std_dev = torch.std(mean_vec_all_langs)

        counter = 0
        outliers = []
        for i in mean_vec_all_langs:
            if abs(i) > mean + args.stdevs * std_dev:
                outliers.append(counter)
            counter += 1

        print(f"Outlier dimensions: {outliers}")
        for o in outliers:
            print(f"Average value of {o}: {mean_vec_all_langs[o]}")

    # ANALYZE ONE AVG EMBEDDING PER LANGUAGE
    elif args.type == "language":
        lang_outliers = defaultdict(lambda: defaultdict())
        all_outliers = set()

        for lang in langs:
            print(f"Considering language {lang}.")
            if type(lang) is tuple:
                lang_or_track = lang[0]
                lang = lang[1]
            else:
                lang_or_track = lang

            target_emb = torch.load(
                f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}{args.append_file_name}.pt')
            mean_vec = torch.mean(target_emb, axis=0)

            mean = torch.mean(mean_vec)
            std_dev = torch.std(mean_vec)

            counter = 0
            outliers = []
            for i in mean_vec:
                if abs(i) > mean + args.stdevs * std_dev:
                    outliers.append(counter)
                counter += 1

            print(f"Outlier dimensions: {outliers}")
            for o in outliers:
                all_outliers.add(o)
                lang_outliers[lang][o] = mean_vec[o]
                print(f"Average value of {o}: {mean_vec[o]}")

        for dim in all_outliers:
            fig_dir = f'../plots/{args.dataset}/{args.model}/{args.layer}/'
            os.makedirs(os.path.dirname(fig_dir), exist_ok=True)
            langs_with_outlier_values = defaultdict()
            # outlier_values = []
            for l in lang_outliers:
                if dim not in lang_outliers[l]:
                    continue
                langs_with_outlier_values[lang_dict_3_2[l]] = lang_outliers[l][dim]
                # langs_with_outlier.append(lang_dict_3_2[l])
                # outlier_values.append(lang_outliers[l][dim])
            langs_with_outlier_values = {k: v for k, v in sorted(langs_with_outlier_values.items(),
                                                                 key=lambda item: item[1])}
            x = np.array(range(0, len(langs_with_outlier_values)))
            y = np.fromiter(langs_with_outlier_values.values(), dtype=float)

            plt.figure(figsize=(7, 1))
            x_ticks = langs_with_outlier_values.keys()
            plt.xticks(x, x_ticks, rotation=90)

            t = [np.around(np.amin(y), 2), np.around(np.amax(y), 2)]
            plt.yticks(t)

            plt.plot(x, y, '.')
            plt.title(f"{dim}")
            plt.savefig(f'{fig_dir}/{dim}_per_lang.png', dpi=300, bbox_inches="tight")
            plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Establish outlier dimensions.')
    parser.add_argument('--model', type=str, help="name of the model")
    parser.add_argument('--layer', type=int, help="which model layer the embeddings are from")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki", 'sts'],
                        help="use embeddings from this dataset (tatoeba, wiki)")
    parser.add_argument('--append_file_name', type=str, default="", help='to load files à la .._whitened.pt')
    parser.add_argument('--stdevs', type=int, default=3,
                        help="dimension must be this number of standard deviations from mean to meet outlier definition")
    parser.add_argument('--type', type=str, help="to analyze per-model or per-language ('all' or 'language')")
    args = parser.parse_args()

    main(args)
