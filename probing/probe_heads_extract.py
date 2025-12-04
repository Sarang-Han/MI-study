# probe_heads_extract.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import torch
import einops
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def parse_layers(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def resolve_layers(user_layers, num_layers):
    out = []
    for L in user_layers:
        L2 = L if L >= 0 else num_layers + L
        if not (0 <= L2 < num_layers):
            raise ValueError(f"Layer index {L} -> {L2} out of range [0, {num_layers-1}]")
        out.append(L2)
    return sorted(set(out))

def last_valid_idx(attn_mask: torch.Tensor, input_ids: torch.Tensor, eos_id: int | None):
    mask = attn_mask.clone().float()
    if mask.size(1) > 0:
        mask[:, 0] = 0.0  # BOS 제외
    if eos_id is not None:
        eos_pos = (input_ids == eos_id)
        mask[eos_pos] = 0.0
    idx = mask.sum(dim=1).long() - 1
    return torch.clamp(idx, min=0)

def make_pre_hook(layer_idx: int, n_q_heads: int, sink: dict):
    def pre_hook(module, inputs):
        x = inputs[0].detach().to("cpu")             # (B,T,E)
        E = x.shape[-1]
        if E % n_q_heads != 0:
            raise RuntimeError(f"[pre_hook] E={E} not divisible by H={n_q_heads}")
        x = einops.rearrange(x, "b t (h d) -> b t h d", h=n_q_heads)  # (B,T,H,D)
        sink[layer_idx] = x
    return pre_hook

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bd2su_llm.csv")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--text-col", default="body_pre")
    ap.add_argument("--label-col", default="cur_su_y")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--layers", default="-1", help="-1 또는 -1,-5,-9")
    ap.add_argument("--batch-size", type=int, default=4)      # ★ 더 보수적으로
    ap.add_argument("--max-length", type=int, default=384)    # ★ 길이도 낮춤
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--body-only", action="store_true")
    ap.add_argument("--out-dir", default="outputs/heads")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.max_samples:
        df = df.head(args.max_samples).copy()

    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    print(f"[info] model_type={getattr(cfg,'model_type',None)} "
          f"num_layers={getattr(cfg,'num_hidden_layers','NA')} "
          f"num_attention_heads={getattr(cfg,'num_attention_heads','NA')} "
          f"num_key_value_heads={getattr(cfg,'num_key_value_heads','NA')}")

    try:
        tok = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=True, local_files_only=True, trust_remote_code=True
        )
    except Exception as e:
        print("[warn] fast tokenizer failed; falling back to slow:", e)
        tok = AutoTokenizer.from_pretrained(
            args.model_path, use_fast=False, local_files_only=True, trust_remote_code=True
        )

    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    ).eval()

    num_layers = len(mdl.model.layers)
    req_layers = parse_layers(args.layers)
    layers = resolve_layers(req_layers, num_layers)
    print(f"[info] requested layers {req_layers} -> resolved {layers} (num_layers={num_layers})")

    # GQA에서도 o_proj 입력의 H는 Q-head 수로 나뉩니다.
    n_q_heads = int(getattr(cfg, "num_attention_heads", 40))

    def build_text(s: str) -> str:
        return s  # 본문만

    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].astype(int).tolist()
    ids = df[args.id_col].tolist()

    X_store = {L: [] for L in layers}
    y_store, id_store = [], []

    # 메모리 단편화 완화(권장)
    torch.backends.cuda.matmul.allow_tf32 = True

    for st in range(0, len(df), args.batch_size):
        batch_txt = [build_text(t) for t in texts[st:st+args.batch_size]]
        enc = tok(batch_txt, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length)
        enc = {k: v.to(mdl.device) for k, v in enc.items()}

        # 훅 등록
        sink, handles = {}, []
        for li, layer in enumerate(mdl.model.layers):
            if li in layers:
                h = layer.self_attn.o_proj.register_forward_pre_hook(
                    make_pre_hook(li, n_q_heads, sink)
                )
                handles.append(h)

        with torch.inference_mode():
            # ★★ 핵심: LM 헤드(로짓) 말고 base만 호출 → 거대 logits 생성 회피
            _ = mdl.model.forward(
                **enc,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False
            )

        if sorted(sink.keys()) != layers:
            print(f"[warn] captured {sorted(sink.keys())}, expected {layers}")

        last_idx = last_valid_idx(enc["attention_mask"], enc["input_ids"], tok.eos_token_id).cpu()

        for L in layers:
            x_bthd = sink[L]                     # (B,T,H,D) on CPU
            sel = [x_bthd[b, int(last_idx[b])] for b in range(x_bthd.size(0))]  # (H,D)
            x_bhd = torch.stack(sel, dim=0)      # (B,H,D)
            X_store[L].append(x_bhd)

        y_store.extend(labels[st:st+args.batch_size])
        id_store.extend(ids[st:st+args.batch_size])

        for h in handles: h.remove()
        del sink, enc
        torch.cuda.empty_cache()

    y_arr = np.asarray(y_store, dtype=np.int64)
    id_arr = np.asarray(id_store)
    for L in layers:
        X = torch.cat(X_store[L], dim=0).float().numpy()  # (N,H,D)
        meta = dict(model=args.model_path, layer=L, token="last", proj="pre_Wo",
                    n_query_heads=n_q_heads)
        out_path = Path(args.out_dir) / f"heads_L{L}_last_preWo.npz"
        np.savez_compressed(out_path, X=X, y=y_arr, ids=id_arr, meta=meta)
        print(f"[saved] {out_path}  X={X.shape} (N,H,D)")
        del X_store[L]

if __name__ == "__main__":
    main()