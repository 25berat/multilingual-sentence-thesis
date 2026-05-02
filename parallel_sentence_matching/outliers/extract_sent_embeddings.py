""" turning the colab file into a script """
import argparse
import os
import shutil
import tarfile
import zipfile

import pandas as pd
import torch
import wget
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from constants import langs_tatoeba, langs_wiki, lang_dict_3_2
from post_processing import whitening, cluster_based
import os
import re
import torch
from pathlib import Path


def read_tatoeba_data(target):
    # read target lang data
    with open(f'../data/tatoeba-parallel/tatoeba.{target}-eng.{target}', 'r', encoding='utf-8') as tgt:
        tgt_sents = [line for line in tgt.read().split("\n") if line]
    # read eng data
    with open(f'../data/tatoeba-parallel/tatoeba.{target}-eng.eng', 'r', encoding='utf-8') as eng:
        eng_sents = [line for line in eng.read().split("\n") if line]
    return tgt_sents, eng_sents


def read_tatoeba_data_task(target):
    """ use the mixed-up order data needed for the tatoeba task """
    target_2 = lang_dict_3_2[target]
    with open(f'../data/tatoeba/{target_2}-en.{target_2}', 'r', encoding='utf-8') as tgt:
        tgt_sents = [line for line in tgt.read().split("\n") if line]
    with open(f'../data/tatoeba/{target_2}-en.en', 'r', encoding='utf-8') as eng:
        eng_sents = [line for line in eng.read().split("\n") if line]
    return tgt_sents, eng_sents


def mean_pooling(token_embeddings, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeds(data, model, tokenizer, args, device):
    tgt_embeddings_layer = None
    attention_masks = None

    dataloader = DataLoader(data, batch_size=args.batch_size, drop_last=False)
    max_len = max([len(x) for x in tokenizer(data, padding=True, truncation=True)['input_ids']])

    for batch in tqdm(dataloader):
        encoded = tokenizer(batch, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
        encoded.to(device)
        if attention_masks is None:
            attention_masks = encoded['attention_mask'].cpu()
        else:
            attention_masks = torch.cat((attention_masks, encoded['attention_mask'].cpu()))
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)
            if tgt_embeddings_layer is None:
                tgt_embeddings_layer = output[2][1:][args.layer].cpu()
            else:
                tgt_embeddings_layer = torch.cat((tgt_embeddings_layer, output[2][1:][args.layer].cpu()))
    return mean_pooling(tgt_embeddings_layer, attention_masks)


def load_model(args, device):
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(device)
    model.eval()
    return model, tokenizer


def sentences_from_two_cols(path):
    col1, col2 = [], []
    with open(path, 'r') as f_in:
        for line in f_in:
            sent1, sent2 = line.strip().split("\t")
            col1.append(sent1)
            col2.append(sent2)
    return col1, col2


def download_bucc(langs):
    os.makedirs('../data/bucc2018/')
    for lg in langs:
        wget.download(f'https://comparable.limsi.fr/bucc2018/bucc2018-{lg}-en.training-gold.tar.bz2',
                      '../data/bucc2018')
        with tarfile.open(f'../data/bucc2018/bucc2018-{lg}-en.training-gold.tar.bz2', 'r:bz2') as tar:
            tar.extractall('../data/')

        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.training.{lg}', f'../data/bucc2018/{lg}-en.test.{lg}')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.training.en', f'../data/bucc2018/{lg}-en.test.en')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.training.gold', f'../data/bucc2018/{lg}-en.test.gold')

        os.remove(f'../data/bucc2018/bucc2018-{lg}-en.training-gold.tar.bz2')

        wget.download(f'https://comparable.limsi.fr/bucc2018/bucc2018-{lg}-en.sample-gold.tar.bz2',
                      '../data/bucc2018')
        with tarfile.open(f'../data/bucc2018/bucc2018-{lg}-en.sample-gold.tar.bz2', 'r:bz2') as tar:
            tar.extractall('../data/')

        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.sample.{lg}', f'../data/bucc2018/{lg}-en.dev.{lg}')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.sample.en', f'../data/bucc2018/{lg}-en.dev.en')
        os.rename(f'../data/bucc2018/{lg}-en/{lg}-en.sample.gold', f'../data/bucc2018/{lg}-en.dev.gold')
        os.remove(f'../data/bucc2018/bucc2018-{lg}-en.sample-gold.tar.bz2')
        os.rmdir(f'../data/bucc2018/{lg}-en/')

def _read_txt(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def extract_bucc_embeds(args, device, langs_unused, model, tokenizer):
    """
    Liest ALLE .txt Dateien in ../data/bucc2018,
    rechnet Embeddings und speichert sie in:
    ../embs/bucc/{model}/{layer}/{language}/emb.pt
    """
    in_dir = Path("../data/bucc2018")
    out_root = Path(f"../embs/bucc/{args.model}/{args.layer}")
    in_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # Sammle alle .txt-Dateien
    txt_files = sorted(in_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Keine .txt Dateien in {in_dir.resolve()} gefunden.")

    print(f"[BUCC] {len(txt_files)} Textdateien gefunden. Starte Embedding-Berechnung …")

    for fpath in txt_files:
        lang_key = fpath.stem  # z.B. deu_Latn
        out_dir = out_root / lang_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "emb.pt"

        # Lade alle Zeilen
        sentences = _read_txt(fpath)
        if not sentences:
            print(f"[WARN] {fpath.name}: Keine gültigen Zeilen gefunden — übersprungen.")
            continue

        # Embeddings berechnen
        print(f"[OK] {fpath.name}: {len(sentences)} Zeilen → rechne Embeddings …")
        embs = get_embeds(sentences, model, tokenizer, args, device)
        if not isinstance(embs, torch.Tensor):
            embs = torch.tensor(embs)

        # Speichern
        torch.save(embs, out_file)
        print(f"[SAVE] {out_file.relative_to(Path('.'))}  (shape={tuple(embs.shape)})")

        # Optional Whitening
        if getattr(args, "save_whitened", False):
            try:
                from post_processing import whitening
                w = whitening(embs.numpy())
                torch.save(torch.tensor(w), out_dir / "emb_whitened.pt")
                print(f"[SAVE] { (out_dir / 'emb_whitened.pt').relative_to(Path('.')) }")
            except Exception as e:
                print(f"[WARN] Whitening fehlgeschlagen für {fpath.name}: {e}")

def main(args):
    try:
        int(args.device)
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    except ValueError:
        device = args.device

    if args.dataset == "tatoeba":
        langs = langs_tatoeba
    elif args.dataset == "wiki":
        langs = langs_wiki
    elif args.dataset == "sts":
        langs = []
    elif args.dataset == 'bucc':
        langs = ['de', 'fr', 'ru', 'zh']
    else:
        raise ValueError("unknown dataset argument")

    model, tokenizer = load_model(args, device)

    if args.dataset == 'tatoeba':
        extract_tatoeba_embeds(args, device, langs, model, tokenizer)

    elif args.dataset == 'wiki':
        extract_wiki_embeds(args, device, langs, model, tokenizer)

    elif args.dataset == 'sts':
        extract_sts_embeds(args, device, model, tokenizer)

    elif args.dataset == 'bucc':
        extract_bucc_embeds(args, device, langs, model, tokenizer)


def download_tatoeba():
    os.makedirs('../data/tatoeba')
    # os.makedirs('../data/tatoeba-parallel')
    os.makedirs('../data/labels/tatoeba')
    wget.download("https://github.com/facebookresearch/LASER/archive/master.zip", "../data/")
    with zipfile.ZipFile('../data/LASER-main.zip', 'r') as zip_ref:
        zip_ref.extractall('../data/')
    os.rename('../data/LASER-main/data/tatoeba/v1/', '../data/tatoeba-parallel/')
    for sl3, sl2 in lang_dict_3_2.items():
        if sl3 == 'eng' or sl3 == 'sun':
            continue
        src_file = f'../data/tatoeba-parallel/tatoeba.{sl3}-eng.{sl3}'
        tgt_file = f'../data/tatoeba-parallel/tatoeba.{sl3}-eng.eng'
        src_out = f'../data/tatoeba/{sl2}-en.{sl2}'
        tgt_out = f'../data/tatoeba/{sl2}-en.en'
        shutil.copy(src_file, src_out)

        tgts = [line.strip() for line in open(tgt_file, 'r', encoding='utf-8')]
        idx = range(len(tgts))
        data = zip(tgts, idx)
        labels_out = open(f'../data/labels/tatoeba/test-{sl2}.tsv', 'w', encoding='utf-8')
        with open(tgt_out, 'w', encoding='utf-8') as ftgt:
            for t, i in sorted(data, key=lambda x: x[0]):
                ftgt.write(f'{t}\n')
                labels_out.write(f'[{i}]\n')
        labels_out.close()
    os.remove('../data/LASER-main.zip')
    shutil.rmtree('../data/LASER-main/')


def extract_tatoeba_embeds(args, device, langs, model, tokenizer):
    if not os.path.isdir('../data/tatoeba'):
        download_tatoeba()
    for lang in langs:
        print(f"Currently processing {lang}...")
        try:
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}')
        except OSError:
            print(f"embeddings for {lang} already exist, skipping")
            continue
        if args.tatoeba_use_task_order:
            tgt, eng = read_tatoeba_data_task(lang)
        else:
            tgt, eng = read_tatoeba_data(lang)
        tgt_embeddings = get_embeds(tgt, model, tokenizer, args, device)
        eng_embeddings = get_embeds(eng, model, tokenizer, args, device)
        torch.save(tgt_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
        torch.save(eng_embeddings, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng.pt')
        if args.save_whitened:
            tgt_whitened = torch.Tensor(whitening(tgt_embeddings.numpy()))
            eng_whitened = torch.Tensor(whitening(eng_embeddings.numpy()))
            torch.save(tgt_whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_whitened.pt')
            torch.save(eng_whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng_whitened.pt')
        if args.save_cbie:
            n_cluster = max(len(eng) // 300, 1)
            tgt_cbie = torch.Tensor(cluster_based(
                tgt_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=tgt_embeddings.shape[1]))
            eng_cbie = torch.Tensor(cluster_based(
                eng_embeddings.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=eng_embeddings.shape[1]))
            torch.save(tgt_cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_cbie.pt')
            torch.save(eng_cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/eng_cbie.pt')
        print(f"Finished saving embeddings for {lang} in model {args.model}.")


def extract_sts_embeds(args, device, model, tokenizer):
    if not os.path.exists('../data/sts'):
        print("Download and extract the STS data.")
        os.makedirs('../data/sts/')
        wget.download("http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.eval.v1.1.zip",
                      '../data/STS_text.zip')
        with zipfile.ZipFile('../data/STS_text.zip', 'r') as zip_ref:
            zip_ref.extractall('../data/sts')

        wget.download("http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.gs.zip", '../data/STS_gt.zip')
        with zipfile.ZipFile('../data/STS_gt.zip', 'r') as zip_ref:
            zip_ref.extractall('../data/sts')

        os.remove('../data/STS_gt.zip')
        os.remove('../data/STS_text.zip')

    def extract_sts_lng(file, track, sentences, lng_id):
        rep = get_embeds(sentences, model, tokenizer, args, device)
        torch.save(rep, f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/{lng_id}.pt')
        if args.save_whitened:
            whitened = torch.Tensor(whitening(rep.numpy()))
            torch.save(whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/{lng_id}_whitened.pt')
        if args.save_cbie:
            n_cluster = max(rep.shape[0] // 300, 1)
            cbie = torch.Tensor(
                cluster_based(rep.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=rep.shape[1]))
            torch.save(cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/{lng_id}_cbie.pt')

    def extract_sts(file, track):
        lng1_sent, lng2_sent = sentences_from_two_cols(f'../data/sts/{file}')
        try:
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{track}/')
        except OSError:
            print(f"embeddings for {track} already exist, skipping")
            return
        extract_sts_lng(file, track, lng1_sent, "lng1")
        extract_sts_lng(file, track, lng2_sent, "lng2")

    print("Extract Track 2 ar-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track2.ar-en.txt', 'track2-ar-en')
    print("Extract Track 4a es-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track4a.es-en.txt', 'track4a-es-en')
    print("Extract Track 4b es-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track4b.es-en.txt', 'track4b-es-en')
    print("Extract Track 6 tr-en.")
    extract_sts('STS2017.eval.v1.1/STS.input.track6.tr-en.txt', 'track6-tr-en')
    print(f"Finished saving STS for model {args.model}.")


def extract_wiki_embeds(args, device, langs, model, tokenizer):
    # Loading Wikipedia datasets
    df_su = pd.read_csv('../data/Wikipedia/Sundanese.csv', sep=',')
    df_sw = pd.read_csv('../data/Wikipedia/Swahili.csv', sep=',')
    df_en = pd.read_csv('../data/Wikipedia/English.csv', sep=',')
    df_es = pd.read_csv('../data/Wikipedia/Spanish.csv', sep=',')
    df_ar = pd.read_csv('../data/Wikipedia/Arabic.csv', sep=',')
    df_tr = pd.read_csv('../data/Wikipedia/Turkish.csv', sep=',')
    dfs = [df_ar, df_en, df_es, df_su, df_sw, df_tr]
    for lang, df in zip(langs, dfs):
        print(f"processing {lang}...")
        try:
            os.makedirs(f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/')
        except OSError:
            print(f"embeddings for {lang} already exist, skipping")
            continue
        rep = get_embeds(df['Sentence'].tolist(), model, tokenizer, args, device)
        torch.save(rep, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}.pt')
        if args.save_whitened:
            whitened = torch.Tensor(whitening(rep.numpy()))
            torch.save(whitened, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_whitened.pt')
        if args.save_cbie:
            n_cluster = max(rep.shape[0] // 300, 1)
            cbie = torch.Tensor(
                cluster_based(rep.numpy(), n_cluster=n_cluster, n_pc=12, hidden_size=rep.shape[1]))
            torch.save(cbie, f'../embs/{args.dataset}/{args.model}/{args.layer}/{lang}/{lang}_cbie.pt')
        print(f"Finished saving embeddings for {lang} in model {args.model}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Establish outlier dimensions.')
    parser.add_argument('--model', type=str, help="name of the model", required=True)
    parser.add_argument('--layer', type=int, help="which model layer to save embeddings are from", required=True)
    parser.add_argument('--device', type=str, default="0", help="which GPU/device to use")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki", "sts", "bucc"],
                        help="which dataset to encode (tatoeba, wiki)")
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'], help='so far only for bucc')
    parser.add_argument('--tatoeba_use_task_order', action='store_true', default=False,
                        help='load tatoeba data with non-parallel order so there is sth to predict')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for encoding')
    parser.add_argument('--save_whitened', action='store_true', help='save embeddings processed with whitening as well')
    parser.add_argument('--save_cbie', action='store_true', help='save embeddings processed with cbie as well')
    args = parser.parse_args()
    main(args)
