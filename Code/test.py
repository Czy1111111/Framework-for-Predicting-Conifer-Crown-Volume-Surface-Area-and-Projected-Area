# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

EPS = 1e-8

TREES_DIR = Path(r"F:\trees")
REAL_DIR = Path(r"F:\treesreal")

MODEL_PATH = TREES_DIR / "joint_model_v6" / "best_joint_model.pth"
OUT_CSV = REAL_DIR / "real_prediction_moe_structure_v7.csv"
ANALYSIS_CSV = REAL_DIR / "real_prediction_moe_analysis_v7.csv"


# ================= utils =================

def clean_name(x):
    return Path(str(x)).stem


def flat(x):

    return np.asarray(x).reshape(-1)


# ================= model =================

class JointModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.emb_head = nn.Linear(512, 256)
        self.scale_head = nn.Linear(512, 1)

        self.geo_head = nn.Sequential(
            nn.Linear(256 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        h = self.encoder(x)
        emb = self.emb_head(h)
        scale = self.scale_head(h)
        return self.geo_head(torch.cat([emb, scale], dim=1))


# ================= physics =================

def physics_prior(df):
    E = df["crown_width_m_E"].values
    W = df["crown_width_m_W"].values
    N = df["crown_width_m_N"].values
    S = df["crown_width_m_S"].values

    hE = df["crown_height_m_E"].values
    hW = df["crown_height_m_W"].values
    hN = df["crown_height_m_N"].values
    hS = df["crown_height_m_S"].values

    V = 0.25 * (E*hE + W*hW + N*hN + S*hS)
    A = 2.0 * (E*hE + W*hW + N*hN + S*hS) * 0.3
    Aproj = np.pi * ((E+W)/2.0) * ((N+S)/2.0)

    return np.stack([V, A, Aproj], axis=1)


# ================= structure =================

def structure_features(df):
    H = df["tree_height_m"].values
    DBH = df["dbh_cm"].values

    E = df["crown_width_m_E"].values
    W = df["crown_width_m_W"].values
    N = df["crown_width_m_N"].values
    S = df["crown_width_m_S"].values

    crown_mean = (E + W + N + S) / 4.0
    crown_asym = np.std(np.stack([E, W, N, S], axis=1), axis=1)

    return np.stack([H, DBH, crown_mean, crown_asym], axis=1)


# ================= MoE =================

class MoE(nn.Module):
    def __init__(self, k=5, in_dim=11):
        super().__init__()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            ) for _ in range(k)
        ])

        self.gate = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.gate(x)
        outs = torch.stack([e(x) for e in self.experts], dim=1)
        w = w.unsqueeze(-1)
        return torch.sum(w * outs, dim=1), w


# ================= metrics =================

def metric(y, p, name=""):
    diff = y - p

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    r2 = 1 - np.sum(diff ** 2) / (np.sum((y - np.mean(y)) ** 2) + EPS)
    rel_bias = np.mean(np.abs(diff) / (np.abs(y) + EPS))

    print(f"\n[{name}]")
    print(f" RMSE     : {rmse:.6f}")
    print(f" MAE      : {mae:.6f}")
    print(f" R2       : {r2:.6f}")
    print(f" RelBias  : {rel_bias:.6f}")

    return rmse, mae, r2, rel_bias


# ================= main =================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_x = pd.read_csv(TREES_DIR / "tree_params.csv")
    df_y = pd.read_csv(TREES_DIR / "training_dataset.csv")

    df_x["filename"] = df_x["filename"].apply(clean_name)
    df_y["filename"] = df_y["filename"].apply(clean_name)

    df = df_x.merge(df_y, on="filename", how="inner")

    BASE_COLS = [
        "tree_height_m","dbh_cm",
        "crown_width_m_E","branch_clearance_m_E","crown_height_m_E",
        "crown_width_m_W","branch_clearance_m_W","crown_height_m_W",
        "crown_width_m_N","branch_clearance_m_N","crown_height_m_N",
        "crown_width_m_S","branch_clearance_m_S","crown_height_m_S"
    ]

    LABEL_COLS = ["V_m3", "A_m2", "Aproj_m2"]

    X = df[BASE_COLS].values.astype(np.float32)
    Y = df[LABEL_COLS].values.astype(np.float32)

    x_mean = X.mean(0)
    x_std = X.std(0) + EPS

    y_log = np.log1p(Y)
    y_mean = y_log.mean(0)
    y_std = y_log.std(0) + EPS

    # ===== real =====
    df_real = pd.read_csv(REAL_DIR / "tree_params_real.csv")
    df_real["filename"] = df_real["filename"].apply(clean_name)

    X_real = df_real[BASE_COLS].values.astype(np.float32)
    Xn = (X_real - x_mean) / x_std

    model = JointModel(len(BASE_COLS)).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        geo = model(torch.from_numpy(Xn).to(device)).cpu().numpy()

    geo = geo[:, :3]

    pred_nn = np.expm1(geo * y_std + y_mean)
    pred_nn = np.clip(pred_nn, 0, None)

    phys = physics_prior(df_real)
    struct = structure_features(df_real)

    X_moe = np.concatenate([pred_nn, phys, struct], axis=1)

    X_tensor = torch.from_numpy(X_moe).float().to(device)
    P_tensor = torch.from_numpy(pred_nn).float().to(device)

    model_moe = MoE(k=5, in_dim=X_moe.shape[1]).to(device)
    opt = optim.Adam(model_moe.parameters(), lr=5e-4)

    for epoch in range(400):
        delta, gate_w = model_moe(X_tensor)
        pred_final = P_tensor + delta

        loss = torch.mean((pred_final - P_tensor) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        delta, gate_w = model_moe(X_tensor)
        pred_final = (P_tensor + delta).cpu().numpy()
        gate_w = gate_w.cpu().numpy()

    gt = df_real[LABEL_COLS].values.astype(np.float32)

    # ================= metrics =================

    for i, n in enumerate(LABEL_COLS):
        metric(gt[:, i], pred_final[:, i], n)

    # ================= save =================

    out = pd.DataFrame({
        "filename": df_real["filename"],
        "V_pred": flat(pred_final[:, 0]),
        "A_pred": flat(pred_final[:, 1]),
        "Aproj_pred": flat(pred_final[:, 2])
    })
    out.to_csv(OUT_CSV, index=False)

    # ================= analysis (FIXED SAFE VERSION) =================

    analysis = pd.DataFrame({
        "filename": flat(df_real["filename"]),

        "V_gt": flat(gt[:, 0]),
        "A_gt": flat(gt[:, 1]),
        "Aproj_gt": flat(gt[:, 2]),

        "V_before": flat(pred_nn[:, 0]),
        "A_before": flat(pred_nn[:, 1]),
        "Aproj_before": flat(pred_nn[:, 2]),

        "V_after": flat(pred_final[:, 0]),
        "A_after": flat(pred_final[:, 1]),
        "Aproj_after": flat(pred_final[:, 2]),

        "err_before": flat(np.mean(np.abs(pred_nn - gt) / (np.abs(gt) + EPS), axis=1)),
        "err_after": flat(np.mean(np.abs(pred_final - gt) / (np.abs(gt) + EPS), axis=1)),

        "crown_mean": flat(struct[:, 2]),
        "crown_asym": flat(struct[:, 3]),
        "volume": flat(gt[:, 0]),

        "phys_V": flat(phys[:, 0]),
        "phys_A": flat(phys[:, 1]),
        "phys_Aproj": flat(phys[:, 2]),

        "expert_id": flat(np.argmax(gate_w, axis=1)),

        "gate_1": flat(gate_w[:, 0]),
        "gate_2": flat(gate_w[:, 1]),
        "gate_3": flat(gate_w[:, 2]),
        "gate_4": flat(gate_w[:, 3]),
        "gate_5": flat(gate_w[:, 4]),
    })

    analysis.to_csv(ANALYSIS_CSV, index=False)

    print("\nSaved:", OUT_CSV)
    print("Saved:", ANALYSIS_CSV)


if __name__ == "__main__":
    main()