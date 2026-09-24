import timm
import torch
import cv2
import pandas as pd
import numpy as np
import torchvision.transforms as T
from wildlife_tools.data import ImageDataset as _BaseImageDataset
from wildlife_tools.train import ArcFaceLoss
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import json
import os
import sys
from PIL import Image

from temporal_sampler import (parse_obs_metadata, TemporalArcFaceSampler,
                              BalancedArcFaceSampler, _DATE_EPOCH)


class ImageDataset(_BaseImageDataset):
    """wildlife_tools ImageDataset that survives unreadable images.
    cv2.imread returns None for corrupt/missing/zero-byte files; the upstream
    get_image then crashes inside cvtColor. We catch that, log the path once,
    and substitute a random other sample so one bad file doesn't kill a run.
    """
    def __getitem__(self, idx):
        for _ in range(10):
            try:
                return super().__getitem__(idx)
            except cv2.error:
                try:
                    path = self.metadata.iloc[idx]['path']
                except Exception:
                    path = f'idx={idx}'
                print(f"[ImageDataset] Unreadable image, substituting: {path}")
                idx = np.random.randint(len(self))
        raise RuntimeError("ImageDataset: 10 consecutive unreadable images")


def unpack_batch(batch):
    """wildlife_tools batches arrive as (image, label) or a dict."""
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    return batch['image'], batch['label']


# ============ Configuration ============
class Config:
    '''
    The script expects the following directory format with unique IDs and observations (roots) separated using subfolders:
    
    Dataset/
        ID1/
            Root1/
                mask1.png
                mask2.png
                ...
            Root2/
                mask1.png
                mask2.png
                ...
        ID2/
            Root3/
                mask1.png
                mask2.png
                ...
            ...
        ...
        
        sorted.csv
    '''
    
    csv_path = "/.. /sorted.csv" # Path to .csv file with images. See GitHub for example csv formatting: https://github.com/jdavidpoling/Coastal-Cod-Individual-Re-Identification
    root = "/.." # Dataset root folder containing .csv and images
    output_dir = "/.." # Output directory for fine-tuned model weights and associated files.
    
    # Model
    model_name = 'hf-hub:BVRA/MegaDescriptor-L-384'
    embedding_size = 1536
    image_size = 384
    
    # Train settings
    batch_size = 16    # used for val/gallery and as P*K reference
    epochs = 20
    lr = 5.1e-6
    weight_decay = 8.7e-3
    warmup_epochs = 1

    # LP-FT (linear-probe then fine-tune; Kumar et al. 2022). Phase 1 trains
    # only the ArcFace head with the backbone frozen, so the head is no longer
    # random when the backbone starts to move; phase 2 fine-tunes at a low LR
    # to limit feature distortion. Aimed at the open-set (oos) degradation.
    # When use_lp_ft is True, `lr`/`warmup_epochs` above are ignored and total
    # training runs for lp_epochs + epochs.
    
    # If use_lp_ft = "True" then this block with override the lr value set above (line ~91).
    use_lp_ft = True # Fine tuning with linear probe (lp).
    lp_epochs = 4            # linear probe duration in epochs
    lp_lr = 1.3e-3             # lp learning rate
    ft_lr = 5.1e-6             # fine-tuning learning rate
    freeze_backbone = False  # True = backbone stays frozen after lp

    
    # !! Temporal PK sampling is a work in progress !! Only enable if you are comfortable adpating the script to how dates are stored in your data.
    
    # Temporal PK sampling: each batch is drawn from a W_days window centred
    # on a uniformly-sampled date in [t_min + W/2, t_max - W/2]. Up to P
    # identities are sampled per batch (K images each). Sparse windows
    # produce smaller batches (down to min_P) instead of being skipped.
    # P=4, K=4 matches the megatune0605 batch shape so this is a pure
    # sampler ablation.
    use_temporal_sampling = False
    
    # Windowless per-identity PK with a draw-time per-observation cap (even
    # split of K across an identity's observations). Isolates balancing from
    # the temporal window — run one sampler or the other, not both. Checked
    # before use_temporal_sampling, so it wins if both are left True.
    use_balanced_sampling = True
    P = 4
    min_P = 2
    K = 4
    W_days = 90

    # Used to ease memory use. Try turning on if you are having OOM issues.
    grad_checkpointing = False

    # Toggle horizontal flips in data augmentation.
    use_horizontal_flip = False

    # Closed-set validation
    #   - val_roots is a list  -> hold out those roots entirely (root-level).
    #   - val_roots is None and val_split_ratio set (e.g. 0.2) -> ID-level
    #     observation split: each identity keeps (1 - ratio) of its
    #     observations for training and holds out the rest for validation.
    #     Identities with < min_obs_for_val observations keep all in train.
    val_split_ratio = None
    min_obs_for_val = 5
    val_roots = ['2299', '1760', '2297', '1516', '933', '2201', '1222', '1049', '2625', '2253', '1599', '2258', '2001', '2784', '2598', '2292', '385', '2006', '1430', '2524', '1585', '2049', '224', '192', '179', '142', '2298', '2156', '1615', '882', '1757', '1968', '1107', '188', '3486', '2074', '1990', '1801', '1677', '1631', '1463', '1345', '1150', '1118', '1052', '410', '165', '2267', '1662', '1422', '1322', '711', '470', '393', '151', '2249', '2210', '1714', '1095', '2288', '2216', '2061', '1713', '153', '1715', '1112', '392', '1843', '1145', '1984', '516', '2269', '877', '1719', '1697', '1159', '1108', '2081', '527', '2531', '1800', '521', '1728', '1051', '2235', '2043', '1775', '1536', '1482', '1412', '1232', '1142', '994', '540', '1722', '1582', '816', '126', '1992', '845', '2261', '2221', '2054', '2033', '1998', '1981', '1789', '1783', '1620', '1538', '1403', '602', '1912', '1034', '965', '938', '842', '358', '2804', '2662', '2414', '438', '1699', '1684', '1666', '1655', '1021', '897', '891', '1824', '2098', '1709', '1014', '2071', '1334', '1716', '898', '1991', '1134', '894', '1020', '2260', '3357', 'HF15102007', 'GBU101002', 'G0SV62004'] #['2064','2049','G0BE31003','GBU101002','1107','1013','1629','1425','1990','1675','1950','1088','1172','1791','1713','1738','1085','1668','1158','1091','1001','1967','1726','1695','1083','1902','1639','1664','1692','1186','1177']


    # !! Work in progress !! Only enable if you are comfortable adpating the script to how dates are stored in your data.
    
    # Optional date range filter (inclusive) applied per split after the
    # root-based split. Dates are parsed from paths via parse_obs_metadata.
    # Format: ('YYYY-MM-DD', 'YYYY-MM-DD') or None for no filter.
    train_date_range = None #("2010-01-01", "2025-01-01")
    val_date_range = None #("2010-01-01", "2025-01-01")
    # If True, rows with no parseable date (alpha-leading filenames, sentinel
    # -1 from parse_obs_metadata) pass through the date filter unchanged.
    keep_undated_in_date_range = True

    # Restrict cross-root validation gallery to identities that also appear
    # in val. Matches baseline_val.py / checkpoint_val.py / megatune0705 so
    # in-training mAP is directly comparable to the 14.8% pretrained baseline
    # and 61.5% previous-ArcFace number.
    filter_gallery_to_shared = True

    # Track-averaged eval: match on the mean embedding of each track's top_n
    # highest-confidence frames (prog_eval-style) instead of per-image, for BOTH the
    # closed-set and open-set metrics. None = use all frames. Off = per-image.
    eval_track_averaging = True
    eval_top_n = None

    # Identity-held-out (open-set) validation: remove these identities from
    # training entirely and use them for an open-set eval.
    # If held_out_identities is a list, use exactly those.
    # If None, auto-pick num_held_out_identities by most distinct roots.
    # Set num_held_out_identities to 0 to disable when held_out_identities is None.
    held_out_identities = ['491105', '491138', '491200', '491206', '635264', '635315', '635352', '635367', 'G0AF41', 'G0AF42', 'G0BE31', 'G0BE91', 'G0BE92', 'G0NG11', 'G0NG12', 'G0SV61', 'G0SV671', 'G0SV672', 'G0V6B21', 'G0V6B22', 'G0VA31', 'G0VA32', 'G0VA61', 'G0VA91', 'G0VA92', 'G0VS11', 'G0VS12', 'GBU111', 'GBU121', 'GBU122', 'HF1121', 'HF1122', 'HF15101', 'HF15911', 'HF15912', 'HF2M01', 'HF2M02', 'HF4A11', 'HF4A12', 'HF4A13', 'HF4H11', 'HF4H12', 'HF6A11', 'HF6A12', 'HF6C11', 'HF6C12', 'HFJ101']
    num_held_out_identities = 0

    # Validation frequency
    val_every_n_epochs = 1

    # Early stopping
    early_stopping_patience = 5  # Stop if no improvement for N epochs

    # Used to set pretrained baseline for visualization only
    frozen_oos_map = 0.2052
    frozen_oos_top1 = 0.3468
    
    # ArcFace settings
    arcface_margin = 0.37
    arcface_scale = 32
    
    # Visualization settings
    num_query_examples = 10
    num_retrieval_results = 5
    num_hardest_examples = 20
    num_gradcam_samples = 10

config = Config()

# Sweep overrides (additive; no env set -> behaves exactly as before)
# sweeps.py launches this script per trial with SWEEP_* env vars. Each maps to a
# Config field and is cast to the field's existing type. Set on the class so the
# instance reads them and config_dict (built from vars(Config)) serialises them.
_sweep_overrides = {
    'SWEEP_LP_LR': ('lp_lr', float),
    'SWEEP_FT_LR': ('ft_lr', float),
    'SWEEP_LP_EPOCHS': ('lp_epochs', int),
    'SWEEP_WEIGHT_DECAY': ('weight_decay', float),
    'SWEEP_ARCFACE_MARGIN': ('arcface_margin', float),
    'SWEEP_ARCFACE_SCALE': ('arcface_scale', int),
    'SWEEP_MODEL_NAME': ('model_name', str),
    'SWEEP_EMBEDDING_SIZE': ('embedding_size', int),
    'SWEEP_IMAGE_SIZE': ('image_size', int),
}
for _env, (_field, _cast) in _sweep_overrides.items():
    if _env in os.environ:
        setattr(Config, _field, _cast(os.environ[_env]))
        print(f"[sweep] {_field} = {getattr(Config, _field)}")

# Config attributes are class-level, so vars(config) (the instance dict) is
# empty; collect the public class attributes for reproducible serialization.
config_dict = {k: v for k, v in vars(Config).items()
               if not k.startswith('_') and not callable(v)}

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Under a sweep each trial gets its own dir; standalone keeps the timestamped one.
run_dir = Path(os.environ.get('SWEEP_RUN_DIR', str(Path(config.output_dir) / f"run_{timestamp}")))
run_dir.mkdir(parents=True, exist_ok=True)
viz_dir = run_dir / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {run_dir}")

# Device Setup
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and prepare data
df = pd.read_csv(config.csv_path)
df = df.rename(columns={'ID': 'identity', 'Path': 'path'})
df['path'] = df['path'].str.replace(config.root + "\\", "", regex=False)
df['path'] = df['path'].str.replace(config.root + "/", "", regex=False)
df['image_id'] = range(len(df))

# Observation = the folder each image lives in (base/identity/observation/img),
# read straight from the path. Robust to filename format and equals the on-disk
# observation folders — the unit the ID-level split and BalancedArcFaceSampler
# operate on.
df['obs_id'] = df['path'].map(lambda p: str(Path(p).parent))
print(f"Observations (image folders): {df['obs_id'].nunique()} across "
      f"{df['identity'].nunique()} identities")
if df['obs_id'].nunique() <= df['identity'].nunique():
    raise ValueError(
        "obs_id collapsed to <= one per identity — the 'path' column may not "
        "include observation folders. Check csv_path and its path format."
    )

# Extract root from filename
def extract_root(path):
    # Root = the observation folder taken from the PATH (base/identity/observation/
    # img), never the filename, so lettered and numeric crops are handled the same
    # way and multi-session fish aren't collapsed by a shared filename token.
    return Path(path).parent.name

df['filename'] = df['path'].apply(lambda x: x.split('/')[-1].split('\\')[-1])
df['root_id'] = df['path'].apply(extract_root).astype(str)  # Always string
# Track = one SAM3 tracklet; only used when eval_track_averaging is on.
def extract_track_key(filename):
    parts = filename.split('_')
    return parts[3] if len(parts) > 3 else parts[0]

def parse_confidence(filename):
    token = os.path.splitext(filename)[0].rsplit('_', 1)[-1]
    digits = ''.join(c for c in token if c.isdigit() or c == '.')
    try:
        return float(digits)
    except ValueError:
        return 1.0

# track_uid = observation path + track key, so a track never crosses observations;
# conf orders frames for the top_n mean (1.0 when a filename carries no conf token).
df['track_uid'] = df['obs_id'] + '__' + df['filename'].map(extract_track_key)
df['conf'] = df['filename'].map(parse_confidence)

print(f"Total dataset: {len(df)} images, {df['identity'].nunique()} unique fish")
print(f"Unique roots: {df['root_id'].nunique()}")
print(f"Root distribution:\n{df['root_id'].value_counts().head(10)}")

# Split by root
unique_roots = df['root_id'].unique()
print(f"\nAll unique roots: {sorted(unique_roots, key=str)}")

if config.val_roots is not None:
    # Root-level: hold out the listed roots entirely.
    val_roots = [str(r) for r in config.val_roots]

    missing_roots = [r for r in val_roots if r not in unique_roots]
    if missing_roots:
        print(f"WARNING: These val_roots not found in data: {missing_roots}")

    train_roots = [str(r) for r in unique_roots if r not in val_roots]

    train_df = df[df['root_id'].isin(train_roots)].reset_index(drop=True)
    val_df = df[df['root_id'].isin(val_roots)].reset_index(drop=True)
else:
    # ID-level observation split. Observation = (identity, counter) from
    # parse_obs_metadata — the same unit BalancedArcFaceSampler trains on, so
    # held-out val observations are never seen during training. Each identity
    # with >= min_obs_for_val observations holds out round(ratio * n_obs) of
    # them (at least 1, leaving at least 1 for train) for validation; the rest
    # train. Identities with fewer observations keep all of theirs in train.
    if config.val_split_ratio is None:
        raise ValueError("val_roots is None, so val_split_ratio must be set "
                         "(e.g. 0.2) for the ID-level observation split.")
    # obs_id (observation = image folder) was built once after data load.

    rng = np.random.default_rng(42)
    val_obs = set()
    for ident, group in df.groupby('identity'):
        obs = group['obs_id'].unique()
        if len(obs) < config.min_obs_for_val:
            continue
        n_val = max(1, min(round(len(obs) * config.val_split_ratio), len(obs) - 1))
        val_obs.update(rng.choice(obs, size=n_val, replace=False))

    val_mask = df['obs_id'].isin(val_obs)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    train_roots = sorted(train_df['root_id'].unique())
    val_roots = sorted(val_df['root_id'].unique())

    print(f"\nID-level observation split (ratio={config.val_split_ratio}, "
          f"min_obs_for_val={config.min_obs_for_val}): "
          f"{df['obs_id'].nunique()} observations across "
          f"{df['identity'].nunique()} identities -> "
          f"{train_df['obs_id'].nunique()} train / {val_df['obs_id'].nunique()} "
          f"val observations; {val_df['identity'].nunique()} identities have "
          f"val observations")

# Optional date range filter (per split; dates parsed from paths)
def apply_date_range(split_df, date_range, name):
    if date_range is None:
        return split_df
    _, dates, _ = parse_obs_metadata(split_df['path'].values)
    start = (datetime.strptime(date_range[0], '%Y-%m-%d') - _DATE_EPOCH).days
    end = (datetime.strptime(date_range[1], '%Y-%m-%d') - _DATE_EPOCH).days
    mask = (dates >= start) & (dates <= end)
    if config.keep_undated_in_date_range:
        mask |= (dates == -1)
    filtered = split_df[mask].reset_index(drop=True)
    print(f"{name} date filter [{date_range[0]} -> {date_range[1]}]: "
          f"{len(split_df)} -> {len(filtered)} images")
    return filtered

train_df = apply_date_range(train_df, config.train_date_range, "Train")
val_df = apply_date_range(val_df, config.val_date_range, "Val")

print(f"\n{'='*50}")
print(f"SPLIT BY ROOT")
print(f"{'='*50}")
print(f"Train roots ({len(train_roots)}): {sorted(train_roots, key=str)}")
print(f"Val roots ({len(val_roots)}):   {sorted(val_roots, key=str)}")
print(f"\nTrain: {len(train_df)} images, {train_df['identity'].nunique()} identities")
print(f"Val:   {len(val_df)} images, {val_df['identity'].nunique()} identities")

# Check for identity overlap (required for cross-root validation)
train_identities = set(train_df['identity'].unique())
val_identities = set(val_df['identity'].unique())
shared_identities = train_identities & val_identities
print(f"\n{'='*50}")
print(f"CROSS-ROOT VALIDATION INFO")
print(f"{'='*50}")
print(f"Identities only in train: {len(train_identities - val_identities)}")
print(f"Identities only in val: {len(val_identities - train_identities)}")
print(f"Shared identities (can evaluate): {len(shared_identities)}")

if len(shared_identities) == 0:
    print("\nWARNING: No shared identities between train and val!")
    print("Cross-root validation will not work.")
    print("Consider a different split strategy.")

# Count evaluatable val images
val_evaluatable = val_df[val_df['identity'].isin(shared_identities)]

if len(val_df) > 0:
    print(f"Evaluatable val images: {len(val_evaluatable)}/{len(val_df)} ({100*len(val_evaluatable)/len(val_df):.1f}%)")
else:
    print("ERROR: No validation images found!")
    print(f"val_roots specified: {val_roots[:5]}...")
    print(f"Available roots in data: {sorted(df['root_id'].unique(), key=str)[:10]}...")
    print("Check that val_roots values match root_id values in your data (all should be strings).")
    exit(1)

# Save split info
split_info = {
    'train_roots': [str(r) for r in train_roots],
    'val_roots': [str(r) for r in val_roots],
    'train_images': len(train_df),
    'val_images': len(val_df),
    'train_identities': int(train_df['identity'].nunique()),
    'val_identities': int(val_df['identity'].nunique()),
    'shared_identities': len(shared_identities),
    'evaluatable_val_images': len(val_evaluatable)
}
with open(run_dir / 'split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

# Unseen/OOS IDs split (open-set validation)
# Their images are split by root into oos_query (one root) + oos_gallery (rest).
oos_query_df = None
oos_gallery_df = None
held_out_identities = []

# Resolve which identities to hold out
if config.held_out_identities is not None:
    requested = [str(i) for i in config.held_out_identities]
    train_id_set = set(train_df['identity'].astype(str))
    missing = [i for i in requested if i not in train_id_set]
    if missing:
        print(f"\nWARNING: held_out_identities not present in train data: {missing}")
    held_out_identities = [i for i in requested if i in train_id_set]

    # Warn about identities that lack >=2 roots (can't form query+gallery)
    train_id_str = train_df.assign(_id=train_df['identity'].astype(str))
    roots_per_id = train_id_str.groupby('_id')['root_id'].nunique()
    insufficient = [i for i in held_out_identities if roots_per_id.get(i, 0) < 2]
    if insufficient:
        print(f"WARNING: these held_out_identities have <2 roots and will be skipped "
              f"in OOS gallery split: {insufficient}")

elif config.num_held_out_identities and config.num_held_out_identities > 0:
    # Auto-pick: identities with the most distinct roots
    roots_per_id = train_df.groupby('identity')['root_id'].nunique().sort_values(ascending=False)
    candidates = roots_per_id[roots_per_id >= 2].index.tolist()
    if len(candidates) < config.num_held_out_identities:
        print(f"\nWARNING: only {len(candidates)} identities have >=2 roots; "
              f"requested {config.num_held_out_identities}. Using all available.")
    held_out_identities = [str(i) for i in candidates[:config.num_held_out_identities]]

if held_out_identities:
    train_id_str = train_df['identity'].astype(str)
    oos_df = train_df[train_id_str.isin(held_out_identities)].reset_index(drop=True)
    train_df = train_df[~train_id_str.isin(held_out_identities)].reset_index(drop=True)

    # Per identity: first root -> query, remaining roots -> gallery
    q_parts, g_parts = [], []
    for ident, group in oos_df.groupby('identity'):
        ident_roots = sorted(group['root_id'].unique())
        if len(ident_roots) < 2:
            continue
        q_root = ident_roots[0]
        q_parts.append(group[group['root_id'] == q_root])
        g_parts.append(group[group['root_id'] != q_root])

    if q_parts and g_parts:
        oos_query_df = pd.concat(q_parts).reset_index(drop=True)
        oos_gallery_df = pd.concat(g_parts).reset_index(drop=True)

        print(f"\n{'='*50}")
        print(f"IDENTITY-HELD-OUT (OPEN-SET) VALIDATION")
        print(f"{'='*50}")
        print(f"Held-out identities: {len(held_out_identities)} ({held_out_identities})")
        print(f"OOS query images:   {len(oos_query_df)}")
        print(f"OOS gallery images: {len(oos_gallery_df)}")
        print(f"Train images after held-out removal: {len(train_df)}")
        print(f"Train identities after held-out removal: {train_df['identity'].nunique()}")

        split_info['held_out_identities'] = [str(i) for i in held_out_identities]
        split_info['oos_query_images'] = len(oos_query_df)
        split_info['oos_gallery_images'] = len(oos_gallery_df)
        split_info['train_images_after_holdout'] = len(train_df)
        split_info['train_identities_after_holdout'] = int(train_df['identity'].nunique())
        with open(run_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
    else:
        print("\nWARNING: no held-out identities had >=2 roots; OOS validation disabled.")
        held_out_identities = []


# Transforms
_train_ops = [
    T.Resize((config.image_size, config.image_size)),
    T.RandomResizedCrop(size=(config.image_size, config.image_size), scale=(0.8, 1.0)),
]
if getattr(config, 'use_horizontal_flip', True):
    _train_ops.append(T.RandomHorizontalFlip())
_train_ops += [
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.3), # Default = 0.3 for all
    T.RandAugment(num_ops=2, magnitude=15),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
]
train_transform = T.Compose(_train_ops)

val_transform = T.Compose([
    T.Resize((config.image_size, config.image_size)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# Datasets & dataLoaders
train_dataset = ImageDataset(train_df, config.root, transform=train_transform)
val_dataset = ImageDataset(val_df, config.root, transform=val_transform)

# Gallery df: optionally restrict to images of identities that also appear in val (removes distractor identities that depress mAP).
if config.filter_gallery_to_shared:
    shared_ids = set(train_df['identity']) & set(val_df['identity'])
    gallery_df = train_df[train_df['identity'].isin(shared_ids)].reset_index(drop=True)
    print(f"Gallery filtered to shared identities only: "
          f"{len(gallery_df):,} images (was {len(train_df):,})")
else:
    gallery_df = train_df

# For validation: train set without augmentation (as gallery)
train_dataset_noaug = ImageDataset(gallery_df, config.root, transform=val_transform)

if config.use_balanced_sampling:
    train_obs = train_df['obs_id'].values
    num_iterations = max(1, len(train_df) // (config.P * config.K))
    train_sampler = BalancedArcFaceSampler(
        labels=train_df['identity'].values,
        counters=train_obs,
        P=config.P, K=config.K,
        num_iterations=num_iterations,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=4, pin_memory=True,
    )
    print(f"\nBalanced PK sampling: P={config.P} x K={config.K}, "
          f"even per-observation split, {num_iterations} iterations/epoch")
elif config.use_temporal_sampling:
    train_counters, train_dates, train_prefixes = parse_obs_metadata(
        train_df['path'].values
    )
    max_batch_size = config.P * config.K
    num_iterations = max(1, len(train_df) // max_batch_size)
    train_sampler = TemporalArcFaceSampler(
        labels=train_df['identity'].values,
        dates=train_dates,
        prefixes=train_prefixes,
        P=config.P, K=config.K, min_P=config.min_P,
        W_days=config.W_days, num_iterations=num_iterations,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )
    print(f"\nTemporal sampling: P<={config.P} (min_P={config.min_P}) x "
          f"K={config.K}, W={config.W_days}d, "
          f"{num_iterations} iterations/epoch")
else:
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

val_loader = DataLoader(
    val_dataset, 
    batch_size=config.batch_size, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

# Gallery loader (train set, no augmentation, for validation)
gallery_loader = DataLoader(
    train_dataset_noaug,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# OOS loaders, if enabled
oos_query_loader = None
oos_gallery_loader = None
if oos_query_df is not None and oos_gallery_df is not None:
    oos_query_dataset = ImageDataset(oos_query_df, config.root, transform=val_transform)
    oos_gallery_dataset = ImageDataset(oos_gallery_df, config.root, transform=val_transform)
    oos_query_loader = DataLoader(
        oos_query_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    oos_gallery_loader = DataLoader(
        oos_gallery_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

# Model
backbone = timm.create_model(
    config.model_name,
    num_classes=0, 
    pretrained=True
).to(device)

if config.grad_checkpointing:
    backbone.set_grad_checkpointing(True)
    print("Gradient checkpointing enabled.")

objective = ArcFaceLoss(
    num_classes=train_dataset.num_classes,
    embedding_size=config.embedding_size,
    margin=config.arcface_margin,
    scale=config.arcface_scale
).to(device)

# Optimizer & scheduler
def set_backbone_trainable(flag):
    for p in backbone.parameters():
        p.requires_grad = flag

if config.use_lp_ft:
    # Phase 1 (linear probe): freeze the backbone, train the ArcFace head only.
    # eval() mode keeps the frozen features deterministic (no dropout).
    set_backbone_trainable(False)
    backbone.eval()
    optimizer = AdamW(objective.parameters(), lr=config.lp_lr,
                      weight_decay=config.weight_decay)
    scheduler = None  # constant LR over the short probe; phase 2 adds cosine
    total_epochs = config.lp_epochs + config.epochs
    print(f"LP-FT enabled: {config.lp_epochs} probe epochs @ lr={config.lp_lr}, "
          f"then {config.epochs} fine-tune epochs @ lr={config.ft_lr}"
          f"{' (backbone stays frozen)' if config.freeze_backbone else ''}")
else:
    optimizer = AdamW(
        list(backbone.parameters()) + list(objective.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    if config.warmup_epochs > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=config.warmup_epochs),
                CosineAnnealingLR(optimizer,
                                  T_max=config.epochs - config.warmup_epochs,
                                  eta_min=1e-6),
            ],
            milestones=[config.warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    total_epochs = config.epochs

scaler = GradScaler(device.type)

# Cross-root validation function
def average_by_track(embeddings, sub_df):
    """Collapse per-image embeddings to L2-normalised per-track means over the top_n
    highest-confidence frames (all frames if eval_top_n is None). sub_df rows are
    positionally aligned to embeddings (validation loaders are shuffle=False)."""
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    embs, ids = [], []
    for _, g in sub_df.groupby('track_uid'):
        rows = g.index.to_numpy()[np.argsort(-g['conf'].to_numpy())]
        if config.eval_top_n:
            rows = rows[:config.eval_top_n]
        m = e[rows].mean(0)
        embs.append(m / (np.linalg.norm(m) + 1e-12))
        ids.append(g['identity'].iloc[0])
    return np.vstack(embs), np.array(ids)


def validate_cross_root(backbone, train_df, val_df, gallery_loader, val_loader, device):
    """
    Cross-root validation: Query (val) vs Gallery (train)
    
    This simulates the real use case:
    - Gallery: known fish database (from training roots)
    - Query: new images from new location (val roots)
    - Task: match query fish to gallery
    
    Only evaluates fish that appear in BOTH train and val sets.
    """
    backbone.eval()
    
    # Get shared identities
    train_identities = set(train_df['identity'].unique())
    val_identities = set(val_df['identity'].unique())
    shared_identities = train_identities & val_identities
    
    if len(shared_identities) == 0:
        print("  WARNING: No shared identities!")
        return {
            'top1_accuracy': 0, 'top5_accuracy': 0, 'mAP': 0,
            'valid_queries': 0, 'total_queries': 0, 'shared_identities': 0
        }
    
    # Extract gallery embeddings (train set)
    gallery_embeddings = []
    gallery_labels = []
    
    with torch.no_grad():
        for batch in tqdm(gallery_loader, desc="  Gallery (train)", leave=False):
            images, labels = unpack_batch(batch)
            
            images = images.to(device)
            with autocast(device.type):
                embeddings = backbone(images)
            
            gallery_embeddings.append(embeddings.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                gallery_labels.append(labels.numpy())
            else:
                gallery_labels.append(np.array(labels))
    
    gallery_embeddings = np.vstack(gallery_embeddings)
    gallery_labels = np.concatenate(gallery_labels)
    gallery_identities = train_df['identity'].values
    
    # Extract query embeddings (val set)
    query_embeddings = []
    query_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Query (val)", leave=False):
            images, labels = unpack_batch(batch)
            
            images = images.to(device)
            with autocast(device.type):
                embeddings = backbone(images)
            
            query_embeddings.append(embeddings.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                query_labels.append(labels.numpy())
            else:
                query_labels.append(np.array(labels))
    
    query_embeddings = np.vstack(query_embeddings)
    query_labels = np.concatenate(query_labels)
    query_identities = val_df['identity'].values
    
    # Optionally collapse per-image embeddings to per-track means (top_n by conf).
    # Covers closed-set and open-set alike since both go through this function.
    if getattr(config, 'eval_track_averaging', False):
        gallery_embeddings, gallery_identities = average_by_track(gallery_embeddings, train_df)
        query_embeddings, query_identities = average_by_track(query_embeddings, val_df)

    # Compute query-to-gallery similarity
    similarity = cosine_similarity(query_embeddings, gallery_embeddings)
    
    # Identify valid queries (fish that exist in gallery)
    valid_query_mask = np.array([qid in shared_identities for qid in query_identities])
    valid_indices = np.where(valid_query_mask)[0]
    
    if len(valid_indices) == 0:
        return {
            'top1_accuracy': 0, 'top5_accuracy': 0, 'mAP': 0,
            'valid_queries': 0, 'total_queries': len(query_identities), 
            'shared_identities': len(shared_identities)
        }
    
    # Rank-1 accuracy
    top1_correct = 0
    for i in valid_indices:
        query_id = query_identities[i]
        top1_idx = np.argmax(similarity[i])
        pred_id = gallery_identities[top1_idx]
        if pred_id == query_id:
            top1_correct += 1
    top1_accuracy = top1_correct / len(valid_indices)
    
    # Rank-5 accuracy
    top5_correct = 0
    for i in valid_indices:
        query_id = query_identities[i]
        top5_indices = np.argsort(similarity[i])[-5:]
        top5_ids = gallery_identities[top5_indices]
        if query_id in top5_ids:
            top5_correct += 1
    top5_accuracy = top5_correct / len(valid_indices)
    
    # Mean average precision (mAP)
    aps = []
    for i in valid_indices:
        query_id = query_identities[i]
        
        # Sort gallery by similarity
        sorted_indices = np.argsort(similarity[i])[::-1]
        sorted_ids = gallery_identities[sorted_indices]
        
        # Find relevant (same identity) in sorted order
        relevant = (sorted_ids == query_id)
        if relevant.sum() == 0:
            continue
        
        # Compute average precision
        precisions = []
        num_relevant = 0
        for j, is_relevant in enumerate(relevant):
            if is_relevant:
                num_relevant += 1
                precisions.append(num_relevant / (j + 1))
        
        if precisions:
            aps.append(np.mean(precisions))
    
    mAP = np.mean(aps) if aps else 0.0
    
    backbone.train()
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'mAP': mAP,
        'valid_queries': len(valid_indices),
        'total_queries': len(query_identities),
        'shared_identities': len(shared_identities)
    }

# Training history
history = {
    'train_loss': [],
    'val_top1': [],
    'val_top5': [],
    'val_mAP': [],
    'oos_top1': [],
    'oos_top5': [],
    'oos_mAP': [],
    'lr': [],
    'valid_queries': [],
    'oos_valid_queries': [],
}

best_mAP = 0.0       # tracked on whichever metric is used for selection
best_metric_name = 'oos_mAP' if oos_query_loader is not None else 'val_mAP'
patience_counter = 0

# Training loop
print("\n" + "="*50)
print("Starting Training")
print("="*50)
print(f"Validation: Cross-root (val queries → train gallery)")
print(f"Early stopping patience: {config.early_stopping_patience} epochs")
print("="*50 + "\n")

for epoch in range(total_epochs):
    # LP-FT phase transition: after lp_epochs of head-only training, unfreeze
    # the backbone (unless freeze_backbone) and switch to the low FT LR with a
    # fresh cosine schedule over the remaining `epochs`. Patience restarts so
    # the fine-tune phase gets a full early-stopping budget.
    if config.use_lp_ft and epoch == config.lp_epochs:
        if config.freeze_backbone:
            print("\n[LP-FT] Phase 2: backbone stays frozen (linear-probe only).")
        else:
            set_backbone_trainable(True)
            print(f"\n[LP-FT] Phase 2: unfreezing backbone, FT lr={config.ft_lr}")
        optimizer = AdamW(
            list(backbone.parameters()) + list(objective.parameters()),
            lr=config.ft_lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
        patience_counter = 0

    in_lp_phase = config.use_lp_ft and epoch < config.lp_epochs
    frozen = in_lp_phase or (config.use_lp_ft and config.freeze_backbone)
    backbone.eval() if frozen else backbone.train()
    running_loss = 0.0

    tag = f"[{'LP' if in_lp_phase else 'FT'}] " if config.use_lp_ft else ""
    pbar = tqdm(train_loader, desc=f"{tag}Epoch {epoch+1}/{total_epochs}")
    
    for batch in pbar:
        images, labels = unpack_batch(batch)
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device.type):
            embeddings = backbone(images)
            loss = objective(embeddings, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_train_loss = running_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    if scheduler is not None:
        scheduler.step()
    
    # Validation (closed and open-set)
    # Validation reads frozen-backbone embeddings, so during the linear-probe
    # phase it can't move off the pretrained baseline — skip eval there and just
    # log the LP train loss; eval (and best-model selection) begins at FT.
    if in_lp_phase:
        print(f"[LP] Epoch {epoch+1}/{total_epochs}  train loss {avg_train_loss:.4f} "
              f"(eval skipped: backbone frozen)")
    elif (epoch + 1) % config.val_every_n_epochs == 0:
        print(f"\nValidating (cross-root: val -> train)...")
        metrics = validate_cross_root(
            backbone, gallery_df, val_df, gallery_loader, val_loader, device
        )

        history['val_top1'].append(metrics['top1_accuracy'])
        history['val_top5'].append(metrics['top5_accuracy'])
        history['val_mAP'].append(metrics['mAP'])
        history['valid_queries'].append(metrics['valid_queries'])

        oos_metrics = None
        if oos_query_loader is not None:
            print(f"Validating (open-set: held-out identities)...")
            oos_metrics = validate_cross_root(
                backbone, oos_gallery_df, oos_query_df,
                oos_gallery_loader, oos_query_loader, device
            )
            history['oos_top1'].append(oos_metrics['top1_accuracy'])
            history['oos_top5'].append(oos_metrics['top5_accuracy'])
            history['oos_mAP'].append(oos_metrics['mAP'])
            history['oos_valid_queries'].append(oos_metrics['valid_queries'])

        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print(f"  Train Loss:     {avg_train_loss:.4f}")
        print(f"  Val Top-1:      {metrics['top1_accuracy']*100:.2f}%")
        print(f"  Val Top-5:      {metrics['top5_accuracy']*100:.2f}%")
        print(f"  Val mAP:        {metrics['mAP']*100:.2f}%")
        print(f"  Valid queries:  {metrics['valid_queries']}/{metrics['total_queries']}")
        if oos_metrics is not None:
            print(f"  OOS Top-1:      {oos_metrics['top1_accuracy']*100:.2f}%")
            print(f"  OOS Top-5:      {oos_metrics['top5_accuracy']*100:.2f}%")
            print(f"  OOS mAP:        {oos_metrics['mAP']*100:.2f}%")
            print(f"  OOS queries:    {oos_metrics['valid_queries']}/{oos_metrics['total_queries']}")
        print(f"  LR:             {optimizer.param_groups[0]['lr']:.6f}")

        # Sweep pruning feed: one flushed line per validation. The driver reads
        # FT-phase lines and reports oos_mAP to Optuna's pruner. Harmless when
        # not sweeping (just an extra small file in run_dir).
        with open(run_dir / 'sweep_metrics.jsonl', 'a') as _mf:
            _mf.write(json.dumps({
                'epoch': epoch,
                'phase': 'LP' if in_lp_phase else 'FT',
                'oos_mAP': (oos_metrics['mAP'] if oos_metrics is not None else None),
                'val_mAP': metrics['mAP'],
            }) + "\n")

        # Best model selection: prefer OOS mAP if available, otherwise val mAP
        current_metric = oos_metrics['mAP'] if oos_metrics is not None else metrics['mAP']
        if current_metric > best_mAP:
            best_mAP = current_metric
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': backbone.state_dict(),
                'objective_state_dict': objective.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'oos_metrics': oos_metrics,
                'best_metric_name': best_metric_name,
            }, run_dir / 'best_model.pt')
            print(f"  ✓ New best model saved! ({best_metric_name}: {best_mAP*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.early_stopping_patience})")

            if patience_counter >= config.early_stopping_patience and not in_lp_phase:
                print(f"\n{'='*50}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"{'='*50}")
                break
    
    # Periodic checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': backbone.state_dict(),
            'objective_state_dict': objective.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, run_dir / f'checkpoint_epoch_{epoch+1}.pt')

# Save final model and results
print("\n" + "="*50)
print("Training Complete!")
print("="*50)

torch.save({
    'model_state_dict': backbone.state_dict(),
    'objective_state_dict': objective.state_dict(),
    'config': config_dict,
}, run_dir / 'final_model.pt')

with open(run_dir / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

with open(run_dir / 'config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"\nBest {best_metric_name}: {best_mAP*100:.2f}%")
print(f"\nResults saved to: {run_dir}")

# Under a sweep, stop here: per-trial gradcam/t-SNE/retrieval plots are wasted
# compute. history.json already holds every metric the driver needs.
if os.environ.get('SWEEP_MODE'):
    sys.exit(0)


# Visualizations
print("\n" + "="*50)
print("Generating Visualizations")
print("="*50 + "\n")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Helper functions
def load_image_for_display(image_path, size=(128, 128)):
    """Load and resize image for display."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        return np.array(img)
    except Exception as e:
        return np.ones((*size, 3), dtype=np.uint8) * 128

def get_full_path(relative_path):
    """Get full path from relative path."""
    return os.path.join(config.root, relative_path)

# Extract final embeddings for visualization
print("Extracting embeddings for visualization...")
backbone.eval()

# Gallery embeddings (train)
gallery_embeddings = []
with torch.no_grad():
    for batch in tqdm(gallery_loader, desc="Gallery embeddings"):
        images, _ = unpack_batch(batch)
        images = images.to(device)
        with autocast(device.type):
            emb = backbone(images)
        gallery_embeddings.append(emb.cpu().numpy())
gallery_embeddings = np.vstack(gallery_embeddings)
gallery_identities = train_df['identity'].values

# Query embeddings (val)
query_embeddings = []
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Query embeddings"):
        images, _ = unpack_batch(batch)
        images = images.to(device)
        with autocast(device.type):
            emb = backbone(images)
        query_embeddings.append(emb.cpu().numpy())
query_embeddings = np.vstack(query_embeddings)
query_identities = val_df['identity'].values

# Cross-root similarity
cross_similarity = cosine_similarity(query_embeddings, gallery_embeddings)

# Viz 1. training curves
print("1. Generating training curves...")
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['val_top1'], 'g-', linewidth=2, label='Val Top-1', marker='o')
    axes[0, 1].plot(history['val_top5'], 'b-', linewidth=2, label='Val Top-5', marker='s')
    if history['oos_top1']:
        axes[0, 1].plot(history['oos_top1'], 'g--', linewidth=2, label='OOS Top-1',
                        marker='o', alpha=0.8)
        axes[0, 1].plot(history['oos_top5'], 'b--', linewidth=2, label='OOS Top-5',
                        marker='s', alpha=0.8)
    if config.frozen_oos_top1 is not None:
        axes[0, 1].axhline(config.frozen_oos_top1, color='gray', linestyle=':', linewidth=2,
                           label=f'Frozen OOS Top-1 ({config.frozen_oos_top1*100:.1f}%)')
    axes[0, 1].set_title('Validation Accuracy (Val vs OOS)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['val_mAP'], 'r-', linewidth=2, marker='o', label='Val mAP')
    if history['oos_mAP']:
        axes[1, 0].plot(history['oos_mAP'], 'm--', linewidth=2, marker='o',
                        alpha=0.8, label='OOS mAP')
    if config.frozen_oos_map is not None:
        axes[1, 0].axhline(config.frozen_oos_map, color='gray', linestyle=':', linewidth=2,
                           label=f'Frozen OOS mAP ({config.frozen_oos_map*100:.1f}%)')
    axes[1, 0].set_title('Validation mAP (Val vs OOS)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmax(history['val_mAP'])
    axes[1, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    axes[1, 0].annotate(f'Best: {history["val_mAP"][best_epoch]*100:.1f}%', 
                        xy=(best_epoch, history['val_mAP'][best_epoch]),
                        xytext=(best_epoch + 1, history['val_mAP'][best_epoch] - 0.05),
                        fontsize=10, color='green')
    
    axes[1, 1].plot(history['lr'], 'purple', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress (Cross-Root Validation)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved training_curves.png")
except Exception as e:
    print(f"   ✗ Training curves failed: {e}")

# Viz 2. t-sne visualization
print("2. Generating t-SNE visualization...")
try:
    from sklearn.manifold import TSNE
    
    # Combine and subsample
    all_embeddings = np.vstack([gallery_embeddings, query_embeddings])
    all_identities = np.concatenate([gallery_identities, query_identities])
    all_sources = np.array(['train'] * len(gallery_embeddings) + ['val'] * len(query_embeddings))
    
    max_points = 2000
    if len(all_embeddings) > max_points:
        indices = np.random.choice(len(all_embeddings), max_points, replace=False)
        emb_subset = all_embeddings[indices]
        id_subset = all_identities[indices]
        source_subset = all_sources[indices]
    else:
        emb_subset = all_embeddings
        id_subset = all_identities
        source_subset = all_sources
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_subset)-1))
    emb_2d = tsne.fit_transform(emb_subset)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot by identity
    unique_ids = np.unique(id_subset)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_ids))))
    
    for i, uid in enumerate(unique_ids[:20]):
        mask = id_subset == uid
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                       c=[colors[i % len(colors)]], alpha=0.6, s=30, label=f'Fish {uid}')
    
    for uid in unique_ids[20:]:
        mask = id_subset == uid
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1], c='gray', alpha=0.3, s=20)
    
    axes[0].set_title('Colored by Fish Identity', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dim 1')
    axes[0].set_ylabel('t-SNE Dim 2')
    
    # Plot by source (train vs val)
    train_mask = source_subset == 'train'
    val_mask = source_subset == 'val'
    
    axes[1].scatter(emb_2d[train_mask, 0], emb_2d[train_mask, 1], 
                   c='blue', alpha=0.5, s=30, label='Train (Gallery)')
    axes[1].scatter(emb_2d[val_mask, 0], emb_2d[val_mask, 1], 
                   c='red', alpha=0.5, s=30, label='Val (Query)')
    
    axes[1].set_title('Colored by Source (Train vs Val)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dim 1')
    axes[1].set_ylabel('t-SNE Dim 2')
    axes[1].legend()
    
    plt.suptitle('t-SNE Embedding Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_dir / 'tsne_embeddings.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved tsne_embeddings.png")
except Exception as e:
    print(f"   ✗ t-SNE visualization failed: {e}")

# Viz 3. cross-root retrieval results
print("3. Generating cross-root retrieval visualization...")
try:
    # Filter to valid queries (shared identities)
    valid_mask = np.array([qid in shared_identities for qid in query_identities])
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        num_queries = min(config.num_query_examples, len(valid_indices))
        query_indices = np.random.choice(valid_indices, num_queries, replace=False)
        
        fig, axes = plt.subplots(num_queries, config.num_retrieval_results + 1, 
                                 figsize=(3 * (config.num_retrieval_results + 1), 3 * num_queries))
        
        if num_queries == 1:
            axes = axes.reshape(1, -1)
        
        for i, query_idx in enumerate(query_indices):
            query_id = query_identities[query_idx]
            query_path = get_full_path(val_df.iloc[query_idx]['path'])
            query_root = val_df.iloc[query_idx]['root_id']
            
            # Get top-K from gallery (train)
            sims = cross_similarity[query_idx]
            top_k_indices = np.argsort(sims)[::-1][:config.num_retrieval_results]
            
            # Plot query
            query_img = load_image_for_display(query_path)
            axes[i, 0].imshow(query_img)
            axes[i, 0].set_title(f'Query (Val)\nID: {query_id}\nRoot: {query_root}', fontsize=9, fontweight='bold')
            axes[i, 0].axis('off')
            axes[i, 0].add_patch(plt.Rectangle((0, 0), query_img.shape[1]-1, query_img.shape[0]-1, 
                                                fill=False, edgecolor='blue', linewidth=3))
            
            # Plot retrieved (from train/gallery)
            for j, ret_idx in enumerate(top_k_indices):
                ret_id = gallery_identities[ret_idx]
                ret_path = get_full_path(train_df.iloc[ret_idx]['path'])
                ret_root = train_df.iloc[ret_idx]['root_id']
                ret_sim = sims[ret_idx]
                
                ret_img = load_image_for_display(ret_path)
                axes[i, j + 1].imshow(ret_img)
                
                is_correct = ret_id == query_id
                color = 'green' if is_correct else 'red'
                symbol = '✓' if is_correct else '✗'
                
                axes[i, j + 1].set_title(f'{symbol} Gallery (Train)\nID: {ret_id}\nRoot: {ret_root}\nSim: {ret_sim:.3f}', 
                                         fontsize=8, color=color)
                axes[i, j + 1].axis('off')
                axes[i, j + 1].add_patch(plt.Rectangle((0, 0), ret_img.shape[1]-1, ret_img.shape[0]-1, 
                                                        fill=False, edgecolor=color, linewidth=3))
        
        plt.suptitle('Cross-Root Retrieval: Val Queries → Train Gallery\n(Different locations)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'cross_root_retrieval.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved cross_root_retrieval.png")
except Exception as e:
    print(f"   ✗ Cross-root retrieval visualization failed: {e}")

# Viz 4. per-root accuracy
print("4. Generating per-root accuracy breakdown...")
try:
    root_accuracies = {}
    root_counts = {}
    
    for idx in range(len(val_df)):
        query_id = query_identities[idx]
        if query_id not in shared_identities:
            continue
        
        root = val_df.iloc[idx]['root_id']
        top1_idx = np.argmax(cross_similarity[idx])
        pred_id = gallery_identities[top1_idx]
        is_correct = (pred_id == query_id)
        
        if root not in root_accuracies:
            root_accuracies[root] = []
            root_counts[root] = 0
        
        root_accuracies[root].append(is_correct)
        root_counts[root] += 1
    
    if root_accuracies:
        roots = list(root_accuracies.keys())
        accuracies = [np.mean(root_accuracies[r]) * 100 for r in roots]
        counts = [root_counts[r] for r in roots]
        
        sorted_indices = np.argsort(accuracies)
        roots = [roots[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(roots) * 0.4)))
        
        bars = ax.barh(range(len(roots)), accuracies, color='steelblue', edgecolor='black')
        
        for bar, acc in zip(bars, accuracies):
            if acc >= 80:
                bar.set_color('green')
            elif acc >= 60:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax.set_yticks(range(len(roots)))
        ax.set_yticklabels([f'Root {r} (n={counts[i]})' for i, r in enumerate(roots)])
        ax.set_xlabel('Cross-Root Top-1 Accuracy (%)')
        ax.set_title('Re-ID Accuracy by Val Root (Query → Train Gallery)', fontsize=14, fontweight='bold')
        ax.axvline(x=np.mean(accuracies), color='black', linestyle='--', label=f'Mean: {np.mean(accuracies):.1f}%')
        ax.legend()
        ax.set_xlim(0, 100)
        
        for i, (acc, count) in enumerate(zip(accuracies, counts)):
            ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'per_root_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved per_root_accuracy.png")
except Exception as e:
    print(f"   ✗ Per-root accuracy visualization failed: {e}")

# Viz 5. hardest negatives (cross-root)
print("5. Generating hardest negatives gallery...")
try:
    hardest_negatives = []
    
    for i in range(len(query_identities)):
        query_id = query_identities[i]
        if query_id not in shared_identities:
            continue
        
        for j in range(len(gallery_identities)):
            gallery_id = gallery_identities[j]
            if gallery_id != query_id:
                hardest_negatives.append((i, j, cross_similarity[i, j], query_id, gallery_id))
    
    hardest_negatives.sort(key=lambda x: x[2], reverse=True)
    
    num_hard = min(config.num_hardest_examples, len(hardest_negatives))
    
    if num_hard > 0:
        fig, axes = plt.subplots(num_hard, 2, figsize=(8, 2.5 * num_hard))
        if num_hard == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (qi, gi, sim, qid, gid) in enumerate(hardest_negatives[:num_hard]):
            query_path = get_full_path(val_df.iloc[qi]['path'])
            gallery_path = get_full_path(train_df.iloc[gi]['path'])
            query_root = val_df.iloc[qi]['root_id']
            gallery_root = train_df.iloc[gi]['root_id']
            
            img1 = load_image_for_display(query_path)
            img2 = load_image_for_display(gallery_path)
            
            axes[idx, 0].imshow(img1)
            axes[idx, 0].set_title(f'Query (Val)\nID: {qid}, Root: {query_root}', fontsize=9)
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(img2)
            axes[idx, 1].set_title(f'Gallery (Train)\nID: {gid}, Root: {gallery_root}', fontsize=9)
            axes[idx, 1].axis('off')
            
            fig.text(0.5, 1 - (idx + 0.5) / num_hard, f'Similarity: {sim:.3f}', 
                     ha='center', fontsize=10, color='red', fontweight='bold',
                     transform=fig.transFigure)
        
        plt.suptitle('Hardest Cross-Root Negatives\n(Different fish with high similarity)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(viz_dir / 'hardest_negatives.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved hardest_negatives.png")
except Exception as e:
    print(f"   ✗ Hardest negatives visualization failed: {e}")

# Viz 6. cmc curve (cross-root)
print("6. Generating CMC curve...")
try:
    valid_mask = np.array([qid in shared_identities for qid in query_identities])
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        max_rank = min(50, len(gallery_identities))
        ranks = range(1, max_rank + 1)
        cmc = []
        
        for rank in ranks:
            correct = 0
            for i in valid_indices:
                query_id = query_identities[i]
                top_k_indices = np.argsort(cross_similarity[i])[::-1][:rank]
                top_k_ids = gallery_identities[top_k_indices]
                if query_id in top_k_ids:
                    correct += 1
            cmc.append(correct / len(valid_indices) * 100)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(ranks, cmc, 'b-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(ranks, cmc, alpha=0.3)
        
        for rank in [1, 5, 10, 20]:
            if rank <= max_rank:
                ax.axvline(x=rank, color='gray', linestyle='--', alpha=0.5)
                ax.annotate(f'Rank-{rank}: {cmc[rank-1]:.1f}%', 
                           xy=(rank, cmc[rank-1]), 
                           xytext=(rank + 2, cmc[rank-1] - 5),
                           fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel('Recognition Rate (%)', fontsize=12)
        ax.set_title('CMC Curve (Cross-Root: Val → Train)', fontsize=14, fontweight='bold')
        ax.set_xlim(1, max_rank)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        summary_text = f'Rank-1: {cmc[0]:.1f}%\nRank-5: {cmc[min(4, len(cmc)-1)]:.1f}%\nRank-10: {cmc[min(9, len(cmc)-1)]:.1f}%'
        ax.text(0.95, 0.05, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'cmc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved cmc_curve.png")
except Exception as e:
    print(f"   ✗ CMC curve visualization failed: {e}")

# Summary
print("\n" + "="*50)
print("All Done!")
print("="*50)
print(f"\nResults saved to: {run_dir}")
print(f"\nModel files:")
print(f"  - best_model.pt ({best_metric_name}: {best_mAP*100:.2f}%)")
print(f"  - final_model.pt")
print(f"\nVisualizations ({viz_dir}):")
print(f"  - training_curves.png")
print(f"  - tsne_embeddings.png")
print(f"  - cross_root_retrieval.png")
print(f"  - per_root_accuracy.png")
print(f"  - hardest_negatives.png")
print(f"  - cmc_curve.png")

# Save final summary
summary = {
    'best_mAP': float(best_mAP),
    'best_metric_name': best_metric_name,
    'validation_type': 'cross-root + identity-held-out' if oos_query_loader is not None else 'cross-root',
    'final_metrics': {
        'val_top1': float(history['val_top1'][-1]) if history['val_top1'] else None,
        'val_top5': float(history['val_top5'][-1]) if history['val_top5'] else None,
        'val_mAP': float(history['val_mAP'][-1]) if history['val_mAP'] else None,
        'oos_top1': float(history['oos_top1'][-1]) if history['oos_top1'] else None,
        'oos_top5': float(history['oos_top5'][-1]) if history['oos_top5'] else None,
        'oos_mAP': float(history['oos_mAP'][-1]) if history['oos_mAP'] else None,
    },
    'dataset': {
        'total_images': len(df),
        'train_images': len(train_df),
        'val_images': len(val_df),
        'shared_identities': len(shared_identities),
        'evaluatable_val_images': len(val_evaluatable),
        'held_out_identities': [str(i) for i in held_out_identities],
        'oos_query_images': len(oos_query_df) if oos_query_df is not None else 0,
        'oos_gallery_images': len(oos_gallery_df) if oos_gallery_df is not None else 0,
    },
    'training': {
        'epochs_completed': len(history['train_loss']),
        'epochs_planned': total_epochs,
        'early_stopped': len(history['train_loss']) < total_epochs,
    }
}

with open(run_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Summary saved to: {run_dir / 'summary.json'}")
