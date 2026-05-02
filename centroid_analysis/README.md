This part of the project is used for centroid-based analysis of multilingual embedding spaces.
It assumes that the embeddings have already been generated beforehand, for example from the matching or mining pipelines.

## Overview

The general workflow is:

1. move or copy the embeddings from the matching or mining pipeline into `/embs`
2. make sure the embeddings are stored under the correct model-specific path
3. for SONAR, run `fix_sonar_tree.py` to convert the embedding file structure
4. run `vis_tsne.py` for centroid visualization
5. run the other centroid and distance calculation scripts


Right now there is only one embedding file in embs/bucc/cis-lmu/glot500-base/7/ace_Arab as a representation. 


## Embedding setup

Before running the centroid analysis, the embeddings must be placed into the `/embs` directory.

The embeddings can come from either:

- parallel sentence matching
- parallel sentence mining

They should be moved or copied into `/embs` under the correct model-specific folder.

Each model has its own path.

Example structure:

- `embs/bucc/cis-lmu/glot500-base/`
- `embs/bucc/laser/`
- `embs/bucc/llama31/`
- `embs/bucc/qwen3/`
- `embs/bucc/sentence-transformers/LaBSE/`
- `embs/bucc/sonar/`
- `embs/bucc/xlm-roberta-base/`

Depending on the model, embeddings may be organized into subfolders such as:

- `7` (raw)
- `cbie`
- `whitened`

For centroid analysis, make sure the embeddings are placed in the correct path for the model you want to analyze.

## Step 1: Move or copy embeddings into `/embs`

Copy or move the embedding files produced by the matching or mining pipeline into the corresponding directory in `/embs`.

Each model uses a unique folder, so the path must match the expected tree.

Examples:

- Glot500: `embs/bucc/cis-lmu/glot500-base/`
- LaBSE: `embs/bucc/sentence-transformers/LaBSE/`
- XLM-R: `embs/bucc/xlm-roberta-base/`
- SONAR: `embs/bucc/sonar/`
- LASER: `embs/bucc/laser/`
- Qwen3: `embs/bucc/qwen3/`
- Llama 3.1: `embs/bucc/llama31/`

Adjust the paths depending on your local setup.

## Step 2: Fix SONAR tree if needed

For SONAR embeddings, run:

python fix_sonar_tree.py

Although the script is called fix_sonar_tree.py, it is not only for SONAR.
The name is just historical. The script can also be used for the other models.

It is used to convert or normalize the embedding file tree into the expected format for the centroid analysis.

Important:

the path inside the script has to be changed manually for each model
you need to set the correct embedding root before running it
run it separately for each model you want to process

So even though the filename says sonar, the script is intended as a general tree-fixing utility for all models.

## Step 3: Run t-SNE visualization

After the embeddings are in the correct directory, run `vis_tsne.py`.

Example:

python vis_tsne.py --dataset bucc --model xlm-roberta-base --layer 7 --emb_dir ../embs/bucc/xlm-roberta-base/7 --emb_name emb.pt --recursive --plot_file ../plots/bucc/xlm-roberta-base/7/MULTI/tsne.png --load torch

Meaning of the main arguments:

- `--model`: model folder name
- `--layer`: embedding layer, for example `7`
- `--dataset`: dataset name, for example `bucc`
- plotfile: where to save the plots

Important:
Adjust the paths in the script if your local embedding directory is different.

Also make sure that:

- the model name matches the folder name
- the layer exists in the embedding tree
- the dataset name matches your data layout
- the selected language is available

## Step 4: Run centroid and distance calculation scripts

After the embeddings are prepared and the visualization works, run the remaining centroid analysis scripts.

These include the scripts for:

- centroid computation
- pairwise centroid distance calculation
- further centroid-based analysis

Run these scripts after checking and adapting the paths to your local setup.

