"""LoRA on Qwen3.5-4B for the reach questions of lora_data.py.

The loss is on the answer letter only: the prompt is the chat template up to the
assistant turn (thinking off, as closed_loop.py asks vLLM), and the letter is
predicted from the last prompt position. Accuracy is the argmax over the option
letters, as closed_loop.py normalises over them. Right padding, so no padded
token precedes a real one (the linear-attention layers are recurrent).

LoRA goes on the language model's attention, linear-attention output and MLP
projections; the vision tower and MTP head are left alone.

Rows with a "soft" label (lora_data.py --suite ml10) train on it: the loss is the
cross-entropy of the letters' probabilities against it, which for a one-hot label is
the usual cross-entropy; accuracy compares the most likely letter with its most likely one.

Under torchrun (WORLD_SIZE > 1) each process takes every WORLD_SIZE-th example of each
batch and the LoRA gradients are summed across processes before the step, which is the
single-GPU step on the whole batch; validation is split the same way. Rank 0 logs and saves.

--init starts from a saved adapter instead of a new one. --lm adds language rows (lang_data.py, kind "lm") to the training rows: their loss is the
mean cross-entropy of the assistant text's tokens (and its end token) after the prompt, and
a batch's loss weighs each row, action or language, the same (--lm-weight scales the latter).

Rows with "images" (lora_data.py --vision) are encoded per batch by the processor,
from the PNG files; text rows are tokenized once up front.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForImageTextToText, AutoProcessor

from tasks import LETTERS

MODEL = "Qwen/Qwen3.5-4B"
TARGET = r".*language_model\.layers\.\d+\..*(q_proj|k_proj|v_proj|o_proj|out_proj|gate_proj|up_proj|down_proj)"


def load(path, tok):
    """(token ids, answer) per text row; ((prompt, image paths), answer) per image row. The answer
    is the letter's index, or the soft label's probabilities per letter."""
    rows = [json.loads(l) for l in open(path)]
    prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for r in rows]
    answers = [r["messages"][2]["content"] + "<|im_end|>" if r.get("kind") == "lm"  # language row: the text
               else tuple(r["soft"]) if "soft" in r else LETTERS.index(r["messages"][2]["content"]) for r in rows]
    if "images" in rows[0]:
        return [((p, r["images"]), a) for p, r, a in zip(prompts, rows, answers)]
    ids = tok(prompts, add_special_tokens=False).input_ids
    return list(zip(ids, answers))


LOADER = ThreadPoolExecutor(8)  # reads PNGs in parallel


def encode(batch, proc, answers=False):
    """The processor's inputs (on the CPU) for image rows: the prompts, with their answer texts when
    answers is True, and the frames."""
    images = list(LOADER.map(lambda f: Image.open(f).convert("RGB"), [f for (_, fs), _ in batch for f in fs]))
    return proc(text=[p + a if answers else p for (p, _), a in batch], images=images,
                padding=True, padding_side="right", add_special_tokens=False, return_tensors="pt")


def batch_logits(model, lm_head, batch, proc, device, enc=None):
    """Next-token logits at the last prompt position of each example (enc: encode()'s, if made already)."""
    if isinstance(batch[0][0], tuple):
        enc = (enc if enc is not None else encode(batch, proc)).to(device)
        last = enc["attention_mask"].sum(1) - 1
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        h = model(**enc).last_hidden_state
        return lm_head(h[torch.arange(len(batch), device=device), last]).float()
    pad = proc.tokenizer.pad_token_id
    n = max(len(x) for x, _ in batch)
    ids = torch.tensor([x + [pad] * (n - len(x)) for x, _ in batch], device=device)
    mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in batch], device=device)
    last = torch.tensor([len(x) - 1 for x, _ in batch], device=device)
    h = model(input_ids=ids, attention_mask=mask).last_hidden_state
    return lm_head(h[torch.arange(len(batch), device=device), last]).float()


def lm_loss(model, lm_head, batch, proc, device, enc=None):
    """Mean over examples of the token cross-entropy of each answer text, given its prompt and frames."""
    tok = proc.tokenizer
    enc = (enc if enc is not None else encode(batch, proc, answers=True)).to(device)
    enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
    h = model(**enc).last_hidden_state
    lengths = enc["attention_mask"].sum(1).tolist()
    losses = []
    for b, ((_, _), answer) in enumerate(batch):
        n = len(tok(answer, add_special_tokens=False).input_ids)
        end = lengths[b]
        logits = lm_head(h[b, end - n - 1:end - 1]).float()  # positions that predict the answer's tokens
        losses.append(F.cross_entropy(logits, enc["input_ids"][b, end - n:end]))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, lm_head, data, letter_ids, proc, device, bs, rank=0, world=1):
    data, correct = data[rank::world], 0
    for i in range(0, len(data), bs):
        batch = data[i:i + bs]
        pred = batch_logits(model, lm_head, batch, proc, device)[:, letter_ids].argmax(-1)
        correct += (pred.cpu() == torch.tensor([a if isinstance(a, int) else int(np.argmax(a)) for _, a in batch])).sum().item()
    counts = torch.tensor([correct, len(data)], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(counts)
    return (counts[0] / counts[1]).item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="out/lora/reach_train.jsonl")
    ap.add_argument("--val", default="out/lora/reach_val.jsonl")
    ap.add_argument("--out", default="out/lora/reach_r16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="split each batch into this many micro-batches")
    ap.add_argument("--lm", default=None, help="language rows (lang_data.py) trained alongside --train")
    ap.add_argument("--lm-weight", type=float, default=1.0)
    ap.add_argument("--init", default=None, help="an adapter to start from (its LoRA config is kept)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)  # the same on every rank, so that all shuffle alike
    torch.manual_seed(args.seed)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    if world > 1:
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    say = print if rank == 0 else (lambda *a, **k: None)

    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    train, val = load(args.train, tok), load(args.val, tok)
    if args.lm:
        train += [x for path in args.lm.split(",") for x in load(path, tok)]
    say(f"train {len(train)}  val {len(val)}  processes {world}")

    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    if args.init:
        model = PeftModel.from_pretrained(model, args.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                                 target_modules=TARGET))
    if rank == 0:
        model.print_trainable_parameters()
    body, lm_head = model.base_model.model.model, model.base_model.model.lm_head

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(train) / args.bs)
    warmup = steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "w") if rank == 0 else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    acc = evaluate(body, lm_head, val, letter_ids, proc, device, 4 * args.bs, rank, world)
    say(f"step 0  val acc {acc:.4f}")
    if log:
        log.write(json.dumps({"step": 0, "val_acc": acc}) + "\n")

    step, t0, loss, loss_lm = 0, time.time(), torch.tensor(0.0), torch.tensor(0.0)
    def prepare(batch):
        """This process's micro-batches of a batch, split into action and language rows, encoded."""
        mine = batch[rank::world]  # this process's share; the loss below is scaled to the whole batch
        micro, out = math.ceil(len(mine) / args.accum), []
        for j in range(0, len(mine), micro):
            part = mine[j:j + micro]
            acts = [x for x in part if not isinstance(x[1], str)]
            texts = [x for x in part if isinstance(x[1], str)]
            image_rows = isinstance(part[0][0], tuple)
            out.append((acts, encode(acts, proc) if acts and image_rows else None,
                        texts, encode(texts, proc, answers=True) if texts else None))
        return out

    prefetch = ThreadPoolExecutor(2)  # encodes the next batches while the GPU works on this one
    for epoch in range(args.epochs):
        random.shuffle(train)
        batches = [train[i:i + args.bs] for i in range(0, len(train), args.bs)]
        ahead = [prefetch.submit(prepare, b) for b in batches[:2]]
        for k, batch in enumerate(batches):
            model.train()
            prepared = ahead.pop(0).result()
            if k + 2 < len(batches):
                ahead.append(prefetch.submit(prepare, batches[k + 2]))
            for acts, enc_a, texts, enc_t in prepared:  # --accum micro-batches, one optimizer step
                total = 0.0
                if acts:
                    logits = batch_logits(body, lm_head, acts, proc, device, enc_a)
                    target = torch.tensor([np.eye(len(letter_ids))[a] if isinstance(a, int) else a for _, a in acts],
                                          dtype=torch.float32, device=device)  # probabilities per letter
                    loss = -(target * F.log_softmax(logits, -1)[:, letter_ids]).sum(-1).mean()
                    total = total + loss * len(acts)
                if texts:
                    loss_lm = lm_loss(body, lm_head, texts, proc, device, enc_t)
                    total = total + args.lm_weight * loss_lm * len(texts)
                (total / len(batch)).backward()
            if world > 1:
                for p in trainable:
                    dist.all_reduce(p.grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                say(f"step {step}/{steps}  loss {loss.item():.4f}"
                    + (f"  lm loss {loss_lm.item():.4f}" if args.lm else "") + f"  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 or step == steps:
                model.eval()
                acc = evaluate(body, lm_head, val, letter_ids, proc, device, 4 * args.bs, rank, world)
                say(f"step {step}  val acc {acc:.4f}", flush=True)
                if log:
                    log.write(json.dumps({"step": step, "loss": loss.item(), "val_acc": acc}) + "\n")
                    log.flush()
    if rank == 0:
        model.save_pretrained(args.out)
        say(f"adapter -> {args.out}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
