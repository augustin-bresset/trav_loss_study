# Traversability Model : study of losses

Currently the traversability problem is constraint by the Positive or Unlabeled biais.
It means that what is labeled is almost always Positive.


## Quickstart

### 1. Installation

```bash
sudo apt-get install libsparsehash-dev
conda env update -f environment.yml
conda activate trav_loss
```

> `torchsparse` is built from source during install — the first run takes a few minutes.

### 2. Build label configs

Generate color/class config files for each dataset (required by the visualizer and training):

```bash
python -m src.preprocessing.build_label_cfg --name goose  --path /data/GOOSE
python -m src.preprocessing.build_label_cfg --name rellis --path /data/RELLIS
```

Output files are saved to `resources/<dataset>_cfg.yaml`.

### 3. Visualize a dataset

```bash
python -m src.visualization --dataset goose  --root /data/GOOSE  --cfg resources/goose_cfg.yaml
python -m src.visualization --dataset rellis --root /data/RELLIS --cfg resources/rellis_cfg.yaml
python -m src.visualization --dataset kitti  --root /data/KITTI  --split val
```

| Key | Action |
|-----|--------|
| `→` / `L` | next frame |
| `←` / `H` | previous frame |
| `R` | reset camera |

Use the left panel to filter classes and inspect per-frame class distributions.

---

## Datasets

Datasets currently supported:

| Dataset | Split arg | Notes |
|---------|-----------|-------|
| [GOOSE 3D](https://goose-dataset.de) | `train` / `val` / `test` | CSV label mapping |
| [RELLIS 3D](https://github.com/unmannedlab/RELLIS-3D) | `train` / `val` / `test` | ontology CSV |
| [SemanticKITTI](http://semantic-kitti.org) | `train` / `val` / `test` | YAML learning map |
| Outback | `train` / `val` / `all` | CSV depth scans |

Labels considered traversable are mapped to positive; all others are treated as unknown (PU learning setting).

