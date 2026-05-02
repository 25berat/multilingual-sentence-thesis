import argparse
import os

from scripts.third_party import evaluate_retrieval


def main(input_args):
    src_langs = ['de', 'fr', 'ru', 'zh']
    args = argparse.Namespace(no_cuda=False, task_name='bucc2018', tgt_language='en', mine_bitext=True,
                              extract_embeds=True, num_layers=12, model_name_or_path=input_args.model_name,
                              tokenizer_name=input_args.tokenizer_name, do_lower_case=False, init_checkpoint=None,
                              batch_size=100, max_seq_length=512, pool_type='cls', pool_skip_special_token=False,
                              embed_size=input_args.dim, model_type=input_args.model_type, use_shift_embeds=False,
                              data_dir=f'{input_args.data_dir}/bucc2018', specific_layer=7, dist='cosine',
                              candidate_prefix='candidates', remove_dim=input_args.remove_dim,
                              predict_dir=f'{input_args.predict_dir}/{input_args.model_name}/bucc2018/')
    for src_lang in src_langs:
        args.src_language = src_lang
        args.log_file = f'mine-bitext-{src_lang}.log'
        out_dir = f'{input_args.out_dir}/bucc2018/{input_args.model_name}-{src_lang}'
        os.makedirs(out_dir, exist_ok=True)
        for split in ['dev', 'test']:
            for lg in [src_lang, 'en']:
                filename = f'{args.data_dir}/{src_lang}-en.{split}.${lg}'
                # from the CLI:
                # cut -f2 $FILE > $OUT/${SL}-${TL}.${sp}.${lg}.txt
                # cut -f1 $FILE > $OUT/${SL}-${TL}.${sp}.${lg}.id
                outfiles = f'{out_dir}/{src_lang}-en.{split}.{lg}'
                with open(filename, 'r', encoding='utf-8') as fin:
                    ids = []
                    text = []
                    for line in fin:
                        id, txt = line.strip().split('\t')
                        ids.append(id)
                        text.append(txt)
                with open(f'{outfiles}.id', 'w+', encoding='utf-8') as idout:
                    for id in ids:
                        idout.write(id + '\n')
                with open(f'{outfiles}.txt', 'w+', encoding='utf-8') as textout:
                    for txt in text:
                        textout.write(txt + '\n')
        args.output_dir = out_dir
        evaluate_retrieval.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model name (HF) or path (local)')
    parser.add_argument('--data_dir', type=str, help='location of bucc2018 data. should include gold files.')
    parser.add_argument('--out_dir', type=str, help='where to write processed (e.g. tokenised) files and embeddings')
    parser.add_argument('--predict_dir', type=str, help='where to save predictions')
    parser.add_argument("--remove_dim", type=int, nargs='*', default=None,
                        help="dimensions to zero out before predicting, i.e. outlier dimensions")
    parser.add_argument('--dim', type=int, default=768)
    args = parser.parse_args()

    args.tokenizer_name = None
    if "bert" in args.model_name:
        args.model_type = "bert"
    if "xlm-roberta" in args.model_name or "sonar" in args.model_name:
        args.model_type = "sonar"
        args.tokenizer_name = "xlm-roberta-base"
    main(args)
