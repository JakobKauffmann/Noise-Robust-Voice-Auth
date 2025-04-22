# scripts/evaluate.py
import argparse, json, os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.raw_dataset import RawAudioPairDataset
from datasets.spectrogram_dataset import MelSpectrogramPairDataset
from models.sincnet import SincNetEmbedding
from models.mobilenet_embedding import MobileNetV2Embedding
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

class FusionClassifier(torch.nn.Module):
    def __init__(self, emb_dim=256, hidden_dim=512):
        super().__init__()
        in_dim=2*2*emb_dim
        self.fc1=torch.nn.Linear(in_dim,hidden_dim)
        self.bn1=torch.nn.BatchNorm1d(hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,hidden_dim//2)
        self.bn2=torch.nn.BatchNorm1d(hidden_dim//2)
        self.out=torch.nn.Linear(hidden_dim//2,1)
    def forward(self,a1,a2,s1,s2):
        e1=torch.cat([a1,s1],dim=1)
        e2=torch.cat([a2,s2],dim=1)
        x=torch.cat([e1,e2],dim=1)
        x=F.relu(self.bn1(self.fc1(x)))
        x=F.relu(self.bn2(self.fc2(x)))
        return torch.sigmoid(self.out(x)).squeeze(1)

def evaluate_split(sinc, spec, clf, raw_csv, spec_csv, batch_size, device):
    raw_ds = RawAudioPairDataset(raw_csv)
    spec_ds= MelSpectrogramPairDataset(spec_csv)
    ds = list(zip(raw_ds, spec_ds))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_y, all_scores = [], []
    sinc.eval(); spec.eval(); clf.eval()
    with torch.no_grad():
        for (w1,w2,y),(i1,i2,_) in loader:
            w1,w2,i1,i2 = w1.to(device),w2.to(device),i1.to(device),i2.to(device)
            y=y.numpy()
            a1,a2 = sinc(w1), sinc(w2)
            s1,s2 = spec(i1), spec(i2)
            scores = clf(a1,a2,s1,s2).cpu().numpy()
            all_y.append(y); all_scores.append(scores)
    y_true = np.concatenate(all_y)
    y_score= np.concatenate(all_scores)
    auc = roc_auc_score(y_true, y_score)
    y_pred = (y_score>=0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/len(y_true)
    fmr = fp/(fp+tn+1e-12)
    fnmr= fn/(fn+tp+1e-12)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1-tpr
    idx = np.nanargmin(np.abs(fnr-fpr))
    eer = (fpr[idx]+fnr[idx])/2
    return {
        "accuracy":acc, "auc":auc, "fmr":fmr,
        "fnmr":fnmr, "eer":eer,
        "confusion":{"tp":int(tp),"tn":int(tn),"fp":int(fp),"fn":int(fn)}
    }

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--fusion_checkpoint",type=Path, required=True)
    p.add_argument("--test_raw", nargs=3, metavar=("NAME","RAW_CSV","SPEC_CSV"),
                   help="repeat for clean, noisy, filtered")
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output_dir",type=Path,default=Path("outputs"))
    p.add_argument("--spec_data_root", type=str, default=None,
               help="Override for .png root")
    return p.parse_args()

def main():
    import argparse, json, os
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from datasets.raw_dataset import RawAudioPairDataset
    from datasets.spectrogram_dataset import MelSpectrogramPairDataset
    from models.sincnet import SincNetEmbedding
    from models.mobilenet_embedding import MobileNetV2Embedding
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

    parser = argparse.ArgumentParser(
        description="Evaluate fusion model on multiple splits"
    )
    parser.add_argument(
        "fusion_checkpoint",
        type=Path,
        help="Path to fusion_best.pt checkpoint"
    )
    parser.add_argument(
        "--test_splits",
        nargs=3,
        action="append",
        metavar=("NAME", "RAW_CSV", "SPEC_CSV"),
        required=True,
        help="One per split: e.g. --test_splits clean data/pairs/pairs_raw_clean.csv data/pairs/pairs_spec_clean.csv"
    )
    parser.add_argument(
        "--spec_data_root",
        type=str,
        default=None,
        help="If set, override PNG paths in CSVs to this local root"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs")
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- load fusion model ---
    sinc = SincNetEmbedding().to(device)
    spec = MobileNetV2Embedding().to(device)
    class FusionClassifier(torch.nn.Module):
        def __init__(self, emb_dim=256, hidden_dim=512):
            super().__init__()
            in_dim = 2 * 2 * emb_dim
            self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
            self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
            self.bn2 = torch.nn.BatchNorm1d(hidden_dim//2)
            self.out = torch.nn.Linear(hidden_dim//2, 1)
        def forward(self, a1, a2, s1, s2):
            e1 = torch.cat([a1, s1], dim=1)
            e2 = torch.cat([a2, s2], dim=1)
            x  = torch.cat([e1, e2], dim=1)
            x  = F.relu(self.bn1(self.fc1(x)))
            x  = F.relu(self.bn2(self.fc2(x)))
            return torch.sigmoid(self.out(x)).squeeze(1)

    clf = FusionClassifier().to(device)
    ckpt = torch.load(args.fusion_checkpoint, map_location=device)
    sinc.load_state_dict(ckpt["sinc"])
    spec.load_state_dict(ckpt["spec"])
    clf.load_state_dict(ckpt["clf"])

    # helper to remap spec paths
    def remap_pairs(pairs, data_root):
        all_in = [p for p,_,_ in pairs] + [q for _,q,_ in pairs]
        orig_root = os.path.commonpath(all_in)
        out = []
        for p1,p2,lbl in pairs:
            rel1 = os.path.relpath(p1, orig_root)
            rel2 = os.path.relpath(p2, orig_root)
            out.append((os.path.join(data_root, rel1),
                        os.path.join(data_root, rel2),
                        lbl))
        return out

    # evaluation loop
    results = {}
    for name, raw_csv, spec_csv in args.test_splits:
        # load datasets
        raw_ds = RawAudioPairDataset(raw_csv)
        spec_ds= MelSpectrogramPairDataset(spec_csv)
        if args.spec_data_root:
            spec_ds.pairs = remap_pairs(spec_ds.pairs, args.spec_data_root)

        # create a DataLoader of zipped pairs
        combined = [(w1,w2,i1,i2,y)
                    for (w1,w2,y),(i1,i2,_) in zip(raw_ds, spec_ds)]
        loader = DataLoader(
            combined,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # run evaluation
        all_labels, all_scores = [], []
        sinc.eval(); spec.eval(); clf.eval()
        with torch.no_grad():
            for w1,w2,i1,i2,y in loader:
                w1,w2,i1,i2 = (t.to(device) for t in (w1,w2,i1,i2))
                y = y.numpy()
                a1 = sinc(w1);   a2 = sinc(w2)
                s1 = spec(i1);   s2 = spec(i2)
                preds = clf(a1,a2,s1,s2).cpu().numpy()
                all_labels.append(y)
                all_scores.append(preds)

        y_true  = np.concatenate(all_labels)
        y_score = np.concatenate(all_scores)

        # compute metrics
        auc = roc_auc_score(y_true, y_score)
        y_pred = (y_score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc  = (tp + tn) / len(y_true)
        fmr  = fp / (fp + tn + 1e-12)
        fnmr = fn / (fn + tp + 1e-12)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2

        results[name] = {
            "accuracy": acc,
            "auc":       auc,
            "fmr":       fmr,
            "fnmr":      fnmr,
            "eer":       eer,
            "confusion": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
        }

    # write JSON
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_file = metrics_dir / "fusion_eval_stats.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)

    # print summary
    print(json.dumps(results, indent=2))

if __name__=="__main__":
    main()
