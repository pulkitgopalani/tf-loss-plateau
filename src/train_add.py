from dotmap import DotMap
import yaml
import argparse

import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity

import wandb
import matplotlib.pyplot as plt

from model_linear import GPTLinear
from data import MultiDigitSum

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(threshold=100000000)

def batch_count(batch_ints, max_len):
    assert len(batch_ints.shape) > 1
    counts = torch.zeros(batch_ints.shape[0], max_len)

    for i in range(batch_ints.shape[0]):
        unpadded_count = torch.bincount(batch_ints[i])
        counts[i, : len(unpadded_count)] = unpadded_count

    return counts

def compute_entropy(seq_pt, max_len):
    eps = 1e-6

    hists = batch_count(seq_pt, max_len=max_len)
    probs = hists / torch.sum(hists, dim=-1, keepdim=True)
    entropy = torch.mean(torch.sum((probs)*torch.log(1/(probs+eps)), dim=-1), dim=0)

    return entropy

def train_step(
    model,
    optim,
    data_sampler,
    step,
    config,
):
    n_train, n_test, num_tokens = (
        config.data.n_train,
        config.data.n_test,
        config.data.num_tokens,
    )

    data = data_sampler.sample(
        num_samples=n_train + n_test,
    )

    train_data = data[:n_train, :]
    test_data = data[n_train:, :]
    
    seq_len = config.data.n
    prompt_len = 2 * seq_len + 2
    gen_len = seq_len + 1
    acc_start = 2 * seq_len + 2

    model.train()
    optim.zero_grad(set_to_none=True)

    _, _, _, loss = model(
        train_data[:, :-1], targets=train_data[:, 1:]
    )
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    model.eval()
    with torch.no_grad():
        # Log train loss, train / test acc, repetition frequency
        attn_map, pre_lm_h, _, train_loss = model(train_data[:, :-1], targets=train_data[:, 1:])
        
        train_pred = model.generate(
            idx=train_data[:, :prompt_len],
            max_new_tokens=gen_len,
        )
        test_pred = model.generate(
            idx=test_data[:, :prompt_len],
            max_new_tokens=gen_len,
        )
                            
        train_acc = torch.mean(
            (train_pred[:, acc_start:] == train_data[:, acc_start:]).to(float)
        ).item()
        test_acc = torch.mean(
            (test_pred[:, acc_start:] == test_data[:, acc_start:]).to(float)
        ).item()

        data_repeat_frac = torch.mean((test_data[:, acc_start:-1] == test_data[:, acc_start+1:]).to(float))
        model_repeat_frac = torch.mean((test_pred[:, acc_start:-1] == test_pred[:, acc_start+1:]).to(float))
        
        # 12 for sequence that output + (10) / = (11) as output digit
        data_entropy = compute_entropy(test_data[:, acc_start:], max_len=12)
        model_entropy = compute_entropy(test_pred[:, acc_start:], max_len=12)

        # Log attention progress measure 
        attn_map_output_seq = attn_map[:, :, acc_start-1:]
        att_mask = torch.zeros_like(attn_map_output_seq).to(device)
        
        att_mask[:, :, 0, config.data.n - 1] = 1
        att_mask[:, :, 0, 2 * config.data.n] = 1

        for output_pos in range(1, gen_len-1):
            att_mask[:, :, output_pos, config.data.n - output_pos - 1] = 1
            att_mask[:, :, output_pos, config.data.n - output_pos] = 1
            att_mask[:, :, output_pos, 2 * config.data.n - output_pos] = 1
            att_mask[:, :, output_pos, 2 * config.data.n - output_pos + 1] = 1
        
        att_mask[:, :, gen_len-1, 0] = 1
        att_mask[:, :, gen_len-1, config.data.n + 1] = 1

        att_prog_measure = torch.mean(
            torch.sum(torch.abs(attn_map_output_seq) * att_mask, dim=(-3, -2, -1)) / 
            torch.sum(torch.abs(attn_map_output_seq), dim=(-3, -2, -1)),
            dim=0
        )

        # Log pair-wise cosine similarity between hidden states
        embed_start = 2 * config.data.n + 1
        embed_len = config.data.n + 1
        logit_cs = torch.zeros((embed_len, embed_len))

        for i_1 in range(embed_start, embed_start + embed_len):
            for i_2 in range(embed_start, i_1):
                logit_cs[i_1 - embed_start, i_2 - embed_start] = torch.mean(
                    (
                        cosine_similarity(
                            pre_lm_h[:, i_1, :], pre_lm_h[:, i_2, :], dim=-1
                        )
                    ),
                    dim=0,
                )

        # Log plots for cosine similarity, attention map
        logit_fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

        im1 = ax[0].imshow(logit_cs)
        ax[0].set_title("avg pre_lm_h cosine sim")
        cb1 = logit_fig.colorbar(im1, location="right", shrink=0.99, pad=0.02, ax=ax[0])

        avg_attn_map = torch.mean(attn_map, dim=0).squeeze().detach().cpu().numpy()
        
        im2 = ax[1].imshow(avg_attn_map)
        ax[1].set_title("att map")
        cb4 = logit_fig.colorbar(im2, location="right", shrink=0.99, pad=0.02, ax=ax[1])
        ax[1].set_xticks(range(avg_attn_map.shape[-1]))
        ax[1].set_yticks(range(avg_attn_map.shape[-2]))

        for i1 in range(embed_len):
            for i2 in range(embed_len):
                text1 = ax[0].text(
                    i2,
                    i1,
                    round(logit_cs[i1, i2].item(), 2),
                    ha="center",
                    va="center",
                    color="w",
                )
    
        print(
            f"Step {step} -- Train loss: {train_loss}, Train Acc: {train_acc} Test Acc: {test_acc}"
        )
        print(f"input: {test_data[0]} \n predicted:{test_pred[0]}")

        if config.train.wandb:

            log_data = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "data_repeat_frac": data_repeat_frac,
                "model_repeat_frac": model_repeat_frac,
                "att_prog_measure": att_prog_measure,
                "pre_lm_h_cosine_sim": logit_fig,
                "data_entropy": data_entropy,
                "model_entropy": model_entropy,
                "mean_cosine_sim": torch.sum(logit_cs[:-1, :]) / (0.5 * (gen_len - 1) * (gen_len - 2))
            }

            for output_pos in range(1, gen_len):
                log_data.update(
                    {
                        f"mean_cosine_sim_{output_pos}": torch.sum(
                            logit_cs[output_pos, :]
                        )
                        / (output_pos)
                    }
                )

            for output_pos in range(gen_len):
                log_data.update(
                        {
                            f"idx_{output_pos}": torch.mean(
                                (
                                    train_pred[:, acc_start + output_pos]
                                    == train_data[:, acc_start + output_pos]
                                ).to(float)
                            ).item()
                        }
                    )

            wandb.log(log_data)

        plt.close()
        del (
            logit_fig,
            ax,
            logit_cs,
        )

        if config.train.save_ckpt:
            if (step == 0) or ((step + 1) % config.train.ckpt_freq == 0):
                model.train()
                torch.save(
                    {
                        "epoch": step,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "train_loss": train_loss,
                        "test_acc": test_acc,
                    },
                    f"./{ckpt_dir}.tar",
                )
                print(f"saved state at epoch {step} to {f'./{ckpt_dir}.tar'}")

                if config.train.wandb:
                    model_wandb = wandb.Artifact(
                        f"model_{ckpt_dir}_step{step}", type="model"
                    )
                    model_wandb.add_file(f"./{ckpt_dir}.tar")
                    wandb.log_artifact(model_wandb)
                    print("model uploaded to wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    config_dir = parser.parse_args().config

    with open(config_dir, "r") as f:
        config = DotMap(yaml.safe_load(f))

    ckpt_dir = str(config_dir).split(".")[0].split("/")[1]

    config.model.vocab_size = 12
    config.model.block_size = 3 * config.data.n + 3
    data_sampler = MultiDigitSum(n=config.data.n, reverse=config.data.reverse)

    model = GPTLinear(config.model, return_att=True).to(device)
    optim = Adam(model.parameters(), lr=config.train.lr)  # , weight_decay=0.08)

    if config.train.wandb:
        wandb_run_name = ckpt_dir
        wandb.login(key="")
        wandb.init(project="", name=wandb_run_name, config=config)
        wandb.watch(model)

    for step in range(config.train.num_steps):
        train_step(
            model=model,
            optim=optim,
            data_sampler=data_sampler,
            step=step,
            config=config,
        )
