import argparse
import json
import os

from constants import langs_tatoeba_2
from scripts.third_party import evaluate_retrieval
from scripts.third_party.evaluate import evaluate


def main(input_args):
    src_langs = langs_tatoeba_2
    args = argparse.Namespace(no_cuda=False, task_name='tatoeba', tgt_language='en', mine_bitext=True,
                              extract_embeds=True, num_layers=12, model_name_or_path=input_args.model_name,
                              tokenizer_name=input_args.tokenizer_name, do_lower_case=False, init_checkpoint=None,
                              batch_size=100, max_seq_length=512, pool_type='cls', pool_skip_special_token=False,
                              embed_size=input_args.dim, model_type=input_args.model_type, use_shift_embeds=False,
                              data_dir=f'{input_args.data_dir}/tatoeba', specific_layer=7, dist='cosine',
                              remove_dim=input_args.remove_dim, extract_rankings=input_args.extract_rankings,
                              predict_dir=f'{input_args.predict_dir}/{input_args.model_name}/tatoeba/')
    out_dir = f'{input_args.out_dir}/tatoeba/{input_args.model_name}_{args.max_seq_length}'
    args.output_dir = out_dir

    for src_lang in src_langs:
        args.src_language = src_lang
        args.log_file = f'tatoeba-{src_lang}.log'
        os.makedirs(out_dir, exist_ok=True)

        evaluate_retrieval.main(args)

    overall_scores, detailed_scores = evaluate(prediction_folder=f'{input_args.predict_dir}/{input_args.model_name}',
                                               label_folder=input_args.labels_dir, verbose=True)
    overall_scores.update(detailed_scores)
    print(json.dumps(overall_scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base', help='model name (HF) or path (local)')
    parser.add_argument('--data_dir', type=str, default='../data/', help='location of tatoeba data')
    parser.add_argument('--labels_dir', type=str, default='../data/labels/', help='where to find labels')
    parser.add_argument('--out_dir', type=str, default='../out',
                        help='where to write processed (e.g. tokenised) files and embeddings')
    parser.add_argument('--predict_dir', default='../predictions',
                        type=str, help='where to save predictions')
    parser.add_argument("--remove_dim", type=int, nargs='*', default=None,
                        help="dimensions to zero out before predicting, i.e. outlier dimensions")
    parser.add_argument('--extract_rankings', default=False, action='store_true', help='flag needed for figure 1')
    parser.add_argument('--dim', type=int, default=768, help='hidden size')
    args = parser.parse_args()

    args.tokenizer_name = None
    if "bert" in args.model_name:
        args.model_type = "bert"
    if "xlm-roberta" in args.model_name or "sonar" in args.model_name:
        args.model_type = "sonar"
        args.tokenizer_name = "xlm-roberta-base"
    main(args)
