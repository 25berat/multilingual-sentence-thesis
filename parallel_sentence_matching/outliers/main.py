import argparse
import json

import numpy as np
import torch
from sklearn.metrics.pairwise import paired_cosine_distances

import anisotropy
import extract_sent_embeddings
import find_outliers
import plots
import vis_tsne
from constants import langs_tatoeba_2, lang_dict, sts_gold_files, langs_wiki, sts_tracks, langs_tatoeba
from scripts.third_party.evaluate import evaluate
from scripts.third_party.evaluate_retrieval import predict_tatoeba


def main():
    models = [#"bert-base-multilingual-uncased"
               "xlm-roberta-base"
              # bert-base-multilingual-cased",
              # "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
              # "../models/xlmr_nliv2_5-langs",
              # "sentence-transformers/LaBSE",
              # "../models/xlmr_across_time/5k"  # requires getting the model conversion to work ://
              # "../models/xlmr_across_time/50k"
              # "../models/xlmr_across_time/500k"
              # "../models/xlmr_across_time/1000k"
              # "../models/xlmr_across_time/1500k"
              ]  # (which) others?

    for model in models:
        args = argparse.Namespace(model=model, layer=7, device='0', dataset='tatoeba', tatoeba_use_task_order=True,
                                  batch_size=64, save_whitened=True, save_cbie=True)
        # extract_analyse(args)

        # checking tatoeba performances
        args = argparse.Namespace(dist='cosine', layer=7, embed_size=768, tgt_language='en')
        # tatoeba(args, 'unmod', model)
        # tatoeba(args, 'cbie', model)
        # tatoeba(args, 'whitened', model)

        # doing similar things for wiki dataset
        args = argparse.Namespace(model=model, layer=11, device='0', dataset='wiki', batch_size=64,
                                  save_whitened=True, save_cbie=True)

        extract_analyse(args)

        # STS
        args = argparse.Namespace(model=model, layer=11, device='0', dataset='sts', batch_size=64,
                                  save_whitened=True, save_cbie=True)
        extract_analyse(args)
        sts(args, 'unmod', model)
        sts(args, 'cbie', model)
        sts(args, 'whitened', model)


def extract_analyse(args):
    extract_sent_embeddings.main(args)
    args.append_file_name = ""
    args.stdevs = 3
    args.type = 'all'

    print("ANALYZING UNMODIFIED EMBEDDINGS (finding outliers)")
    find_outliers.main(args)
    args.dim = -1
    args.verbose = False
    print("ANALYZING UNMODIFIED EMBEDDINGS (anisotropy)")
    anisotropy.main(args)
    args.job = 'means'
    args.language = ''
    print("PLOTTING UNMODIFIED EMBEDDINGS")
    plots.main(args)
    # commented out because it takes a long time to do for all the cases
    # tsne_vis(args)

    # cbie version
    args.append_file_name = "_cbie"
    print("ANALYZING CBIE EMBEDDINGS (finding outliers)")
    find_outliers.main(args)
    print("ANALYZING CBIE EMBEDDINGS (anisotropy)")
    anisotropy.main(args)
    print("PLOTTING CBIE EMBEDDINGS")
    plots.main(args)
    # tsne_vis(args)

    # whitening version
    args.append_file_name = "_whitened"
    print("ANALYZING WHITENED EMBEDDINGS (finding outliers)")
    find_outliers.main(args)
    print("ANALYZING WHITENED EMBEDDINGS (anisotropy)")
    anisotropy.main(args)
    print("PLOTTING WHITENED EMBEDDINGS")
    plots.main(args)
    # tsne_vis(args)


def tsne_vis(args):
    args.do_whiten = False
    args.do_cbie = False
    args.load = 'torch'
    args.parallel_vis = args.dataset in ['tatoeba', 'sts']
    args.emb_file = None
    args.parallel_emb_file = None
    args.plot_file = None
    langs = langs_tatoeba if args.dataset == 'tatoeba' else langs_wiki if args.dataset == 'wiki' else sts_tracks
    for lang_or_track in langs:
        args.lang_or_track = lang_or_track
        args.emb_file = None
        args.parallel_emb_file = None
        args.plot_file = None
        vis_tsne.main(args)


def tatoeba(args, post_proc, model):
    print(f"PREDICTING TATOEBA FOR {post_proc} EMBEDDINGS")
    args.predict_dir = f'../predictions/{model}/{post_proc}/tatoeba/'
    append = '' if post_proc == 'unmod' else f'_{post_proc}'
    for lang in langs_tatoeba_2:
        args.src_language = lang
        lang_3 = lang_dict[lang]
        src = torch.load(f'../embs/tatoeba/{model}/{args.layer}/{lang_3}/{lang_3}{append}.pt').numpy().astype(
            np.float32)
        tgt = torch.load(f'../embs/tatoeba/{model}/{args.layer}/{lang_3}/eng{append}.pt').numpy().astype(np.float32)
        predict_tatoeba(args, src, tgt)
    overall_scores, detailed_scores = evaluate(
        prediction_folder=f'../predictions/{model}/{post_proc}/', label_folder='../data/labels/'
    )
    overall_scores.update(detailed_scores)
    print(json.dumps(overall_scores))


STS_GOLD = {}


def _load_float_file(path):
    with open(path) as f_in:
        float_list = [float(line.strip()) for line in f_in]
    return np.array(float_list)


def _load_sts_gold():
    STS_GOLD.update({
        task: _load_float_file(path)
        for task, path in sts_gold_files.items()
    })


def sts(args, post_proc, model):
    print(f"PREDICTING STS FOR {post_proc} EMBEDDINGS")
    if not STS_GOLD:
        _load_sts_gold()

    args.predict_dir = f'../predictions/{model}/{post_proc}/tatoeba/'
    append = '' if post_proc == 'unmod' else f'_{post_proc}'

    scores = {}
    for track, true_sim in STS_GOLD.items():
        lng1 = torch.load(f'../embs/sts/{model}/{args.layer}/{track}/lng1{append}.pt').numpy().astype(np.float32)
        lng2 = torch.load(f'../embs/sts/{model}/{args.layer}/{track}/lng2{append}.pt').numpy().astype(np.float32)
        model_sim = 1 - paired_cosine_distances(lng1, lng2)
        scores[track] = np.corrcoef(np.stack((model_sim, true_sim)))[0, 1]

    print(json.dumps(scores))


if __name__ == "__main__":
    main()
