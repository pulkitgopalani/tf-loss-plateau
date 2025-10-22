import yaml
import argparse
from dotmap import DotMap
import math

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import cosine_similarity

import wandb
import matplotlib.pyplot as plt

from model_linear import GPTLinear
from data import MovingWindowSum

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(threshold=100000000)


def linear_warmup(warmup_steps):
    # https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/optimization.py#L72
    def warmup_func(step):
        if step < warmup_steps:
            return float(step) / float(max(1.0, warmup_steps))
        return 1.0

    return warmup_func


def linear_warmup_cosine_decay(warmup_steps, total_steps):
    # https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/optimization.py#L132

    def warmup_cosine_func(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return warmup_cosine_func


def train_step(
    model,
    optim,
    lr_schedule,
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
        num_tokens=num_tokens,
    )

    train_data = data[:n_train, :]
    test_data = data[n_train:, :]

    prompt_len = num_tokens + 1
    gen_len = num_tokens
    acc_start = num_tokens + 1

    model.train()
    optim.zero_grad(set_to_none=True)

    _, _, _, loss = model(train_data[:, :-1], targets=train_data[:, 1:])
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()
    lr_schedule.step()

    model.eval()
    with torch.no_grad():
        # Log train loss, train / test acc, repetition frequency
        attn_map, pre_lm_h, _, train_loss = model(
            train_data[:, :-1], targets=train_data[:, 1:]
        )

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

        model_repeat_frac = torch.mean(
            (test_pred[:, acc_start:-1] == test_pred[:, acc_start + 1 :]).to(float)
        )

        # Log attention progress measure
        attn_map_output_seq = attn_map[:, :, acc_start - 1 :]
        att_mask = torch.zeros_like(attn_map_output_seq).to(device)

        att_mask[:, :, 0, 0] = 1
        for i in range(num_tokens - 1):
            att_mask[:, :, i + 1, i : i + 2] = 1

        att_prog_measure = torch.mean(
            torch.sum(torch.abs(attn_map_output_seq) * att_mask, dim=(-3, -2, -1))
            / torch.sum(torch.abs(attn_map_output_seq), dim=(-3, -2, -1)),
            dim=0,
        )

        # Log pair-wise cosine similarity between hidden states
        embed_start = acc_start - 1
        embed_len = gen_len

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

        if step % 100 == 0:
            # Log plots for cosine similarity, attention map
            logit_fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

            im1 = ax[0].imshow(logit_cs)
            ax[0].set_title("avg pre_lm_h cosine sim")
            cb1 = logit_fig.colorbar(
                im1, location="right", shrink=0.99, pad=0.02, ax=ax[0]
            )

            avg_attn_map = torch.mean(attn_map, dim=0).squeeze().detach().cpu().numpy()

            im2 = ax[1].imshow(avg_attn_map)
            ax[1].set_title("att map")
            cb4 = logit_fig.colorbar(
                im2, location="right", shrink=0.99, pad=0.02, ax=ax[1]
            )
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
        # print(f"input: {test_data[0]} \n predicted:{test_pred[0]}")

        if config.train.wandb:

            log_data = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                # "data_repeat_frac": data_repeat_frac,
                "lr": optim.param_groups[0]["lr"],
                "model_repeat_frac": model_repeat_frac,
                "att_prog_measure": att_prog_measure,
                "mean_cosine_sim": torch.sum(logit_cs[:, 1:])
                / (0.5 * (gen_len - 1) * (gen_len - 2)),
            }

            for output_pos in range(gen_len):
                log_data.update(
                    {
                        f"idx{output_pos}_check": torch.mean(
                            (
                                train_pred[:, acc_start + output_pos]
                                == train_data[:, acc_start + output_pos]
                            ).to(float)
                        ).item()
                    }
                )

                if output_pos < gen_len - 1:
                    log_data.update(
                        {
                            f"mean_cosine_sim_{output_pos}": torch.sum(
                                logit_cs[:, output_pos]
                            )
                            / (gen_len - 1 - output_pos)
                        }
                    )

            if step % 100 == 0:
                log_data.update({"pre_lm_h_cosine_sim": logit_fig})

            wandb.log(log_data)

        if step % 100 == 0:
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
                    "./mws_k2_l1_h1_a16_n16.tar",
                )
                print(f"saved state at epoch {step} to {f'./mws_k2_l1_h1_a16_n16.tar'}")

                if config.train.wandb:
                    model_wandb = wandb.Artifact(f"model_step{step}", type="model")
                    model_wandb.add_file(f"./mws_k2_l1_h1_a16_n16.tar")
                    wandb.log_artifact(model_wandb)
                    print("model uploaded to wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    config_dir = parser.parse_args().config

    with open(config_dir, "r") as f:
        config = DotMap(yaml.safe_load(f))

    ckpt_dir = str(config_dir).split(".")[0].split("/")[1]

    config.model.vocab_size = max(config.data.p, config.data.max_num) + 1
    config.model.block_size = 2 * config.data.num_tokens + 1

    data_sampler = MovingWindowSum(
        min_num=config.data.min_num,
        max_num=config.data.max_num,
        k=config.data.k,
        p=config.data.p,
    )

    model = GPTLinear(config.model, return_att=True).to(device)
    optim = Adam(model.parameters(), lr=config.train.lr)

    if config.train.lr_schedule == "linear":
        lr_schedule_func = linear_warmup(warmup_steps=config.train.warmup_steps)
    elif config.train.lr_schedule == "linear_cosine":
        lr_schedule_func = linear_warmup_cosine_decay(
            warmup_steps=config.train.warmup_steps, total_steps=config.train.num_steps
        )

    lr_schedule = LambdaLR(optimizer=optim, lr_lambda=lr_schedule_func)

    if config.train.wandb:
        wandb_run_name = (
            f"mws_adam_{config.train.lr_schedule}_{config.train.warmup_steps}"
        )
        wandb.login(key="")
        wandb.init(project="tf-emergence", name=wandb_run_name, config=config)
        wandb.watch(model)

    for step in range(config.train.num_steps):
        train_step(
            model=model,
            optim=optim,
            lr_schedule=lr_schedule,
            data_sampler=data_sampler,
            step=step,
            config=config,
        )
