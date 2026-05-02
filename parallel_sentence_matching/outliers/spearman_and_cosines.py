import argparse
import json
from scipy import stats
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from constants import langs_tatoeba_2

parser = argparse.ArgumentParser(description='Analyze prediction rankings and cosines.')
parser.add_argument('--model', type=str, help="name of the model to be analyzed")
parser.add_argument('--dimension', type=int, help='dimension to analyze, e.g. 588')
parser.add_argument('--job', help="'spearman' or 'cosines")
args = parser.parse_args()

langs = langs_tatoeba_2

all_correlations = defaultdict(float)


def spearman_per_sentence(original_sent, modified_sent):
    counter = 1
    original_ranking = defaultdict()
    modified_ranking = defaultdict()

    for pred in original_sent:
        original_ranking[pred] = counter
        counter += 1
    for mod in modified_sent:
        modified_ranking[mod] = original_ranking.get(mod, 0)

    orig_ranking = list(original_ranking.values())
    mod_ranking = list(modified_ranking.values())
    # handle possible 'correlation coefficient is not defined' error
    # happens if all integers in the list are the same, i.e. 0 in this case
    # (would happen if all predictions in mod are different from orig)
    if all(p == 0 for p in mod_ranking):
        mod_ranking[-1] = -1
    return stats.spearmanr(orig_ranking, mod_ranking)


def calculate_spearman():
    for lang in langs:
        correlation_sum = 0
        with open(f"../predictions/{args.model}-original-predictions/tatoeba/ids/test-{lang}.json") as c:
            with open(f"../predictions/{args.model}-{args.dimension}-predictions/tatoeba/ids/test-{lang}.json") as f:
                original = json.load(c)
                modified = json.load(f)

                for i in range(len(original)):
                    rho, pval = spearman_per_sentence(original[i][:10], modified[i][:10])
                    # rho, pval = spearman_per_sentence(original[i], modified[i])
                    correlation_sum += rho
        all_correlations[lang] = correlation_sum / len(original)

    all_correlations['avg'] = sum(all_correlations.values()) / len(all_correlations)
    print(all_correlations)


def cosine_difference():
    matplotlib.rcParams.update({'font.size': 12})
    colors = ["#F06292", "#880E4F", "#FF80AB", "#CE93D8", "#9575CD", "#8C9EFF", "#3F51B5", "#90CAF9",
              "#1E88E5", "#1565C0", "#82B1FF", "#29B6F6", "#01579B", "#0097A7", "#4DB6AC", "#00695C",
              "#81C784", "#43A047", "#2E7D32", "#AED581", "#689F38", "#FFEE58", "#FBC02D", "#F57F17",
              "#EF6C00", "#FFD180", "#FF9E80", "#BF360C", "#E64A19", "#795548", "#EA80FC", "#BA68C8", "#E1BEE7",
              "#F3E5F5"]
    color_id = 0
    for lang in langs:
        if lang == "jv" or lang == "te":
            continue
        with open(f"../predictions/{args.model}-original-predictions/tatoeba/cosines/test-cosines-{lang}.json") as c:
            with open(f"../predictions/{args.model}-{args.dimension}-predictions/tatoeba/cosines/test-cosines-{lang}.json") as f:
                original = json.load(c)
                modified = json.load(f)
                original = original
                modified = modified
                for i in range(0, len(original) - 1):
                    original[i + 1] = np.add(np.array(original[i]), np.array(original[i + 1]))
                    modified[i + 1] = np.add(np.array(modified[i]), np.array(modified[i + 1]))
                orig_mean_cosines = np.divide(original[-1], len(original))
                mod_mean_cosines = np.divide(modified[-1], len(modified))
                # plt.plot(orig_mean_cosines[:3], color=colors[color_id], label=lang)
                # plt.plot(mod_mean_cosines[:3], linestyle="--", color=colors[color_id], label='_nolegend_')
                plt.plot(orig_mean_cosines, color=colors[color_id], label=lang)
                plt.plot(mod_mean_cosines, linestyle="--", color=colors[color_id], label='_nolegend_')
                color_id += 1
    ax = plt.gca()
    ax.set_ylim([0, 0.95])
    ax.set_xlim([-5, 255])
    plt.legend(title="\u2014  original\n---  modified\n", ncol=2, loc='center left',
               bbox_to_anchor=(1, 0.5), handlelength=0.8, columnspacing=0.5, borderpad=0.0,
               labelspacing=0.05, title_fontsize=18, frameon=False)
    plt.xlabel("ranking position", fontsize=18)
    plt.xticks(np.arange(0, 251, 50))
    plt.ylabel("cosine distance", fontsize=18)
    plt.savefig(f'../plots/tatoeba/cosines_{args.model}_{args.dimension}.pdf', bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    if args.job == "spearman":
        print("Selected Spearman.")
        calculate_spearman()
    elif args.job == "cosines":
        print("Selected Cosines.")
        cosine_difference()
