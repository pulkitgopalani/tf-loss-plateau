import yaml
import argparse
from dotmap import DotMap

import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity

import wandb
import matplotlib.pyplot as plt

from model_linear import GPTLinear
from data import ConcatMovingWindowSum

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(threshold=100000000)


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
        num_tokens=num_tokens,
    )

    train_data = data[:n_train, :]
    test_data = data[n_train:, :]

    prompt_len = num_tokens + 1
    gen_len = train_data.shape[1] - prompt_len
    acc_start = num_tokens + 1

    model.train()
    optim.zero_grad(set_to_none=True)

    _, _, _, loss = model(train_data[:, :-1], targets=train_data[:, 1:])
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    model.eval()
    with torch.no_grad():

        # Log train loss, train / test acc, repetition frequency
        attn_map, _, _, test_loss = model(
            test_data[:, :-1], targets=test_data[:, 1:]
        )

        _, pre_lm_h, _, train_loss = model(
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

        print(f"Step {step} -- Test loss: {test_loss}, Test Acc: {test_acc}")
        print(f"input: {test_data[0]} \n predicted:{test_pred[0]}")

        # Log plots for cosine similarity, attention map
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

        cosine_sim_1 = torch.sum(logit_cs[:, :4]) / 42
        cosine_sim_2 = torch.sum(logit_cs[:, 4:9]) / 30
        cosine_sim_3 = torch.sum(logit_cs[:, 9:]) / 6

        train_acc_1 = torch.mean(
            (train_pred[:, acc_start:acc_start+4] == train_data[:, acc_start:acc_start+4]).to(float)
        ).item()

        train_acc_2 = torch.mean(
            (train_pred[:, acc_start+4:acc_start+9] == train_data[:, acc_start+4:acc_start+9]).to(float)
        ).item()

        train_acc_3 = torch.mean(
            (train_pred[:, acc_start+9:] == train_data[:, acc_start+9:]).to(float)
        ).item()

        if config.train.wandb:

            log_data = {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "cosine_sim_1": cosine_sim_1,
                "cosine_sim_2": cosine_sim_2,
                "cosine_sim_3": cosine_sim_3,
                "train_acc_1": train_acc_1,
                "train_acc_2": train_acc_2,
                "train_acc_3": train_acc_3,
            }


            if step % 10 == 0:

                logit_fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

                im1 = ax[0].imshow(logit_cs)
                ax[0].set_title("avg pre_lm_h cosine sim")
                cb1 = logit_fig.colorbar(
                    im1, location="right", shrink=0.99, pad=0.02, ax=ax[0]
                )

                avg_attn_map = (
                    torch.mean(attn_map, dim=0).squeeze().detach().cpu().numpy()
                )

                im2 = ax[1].imshow(avg_attn_map)
                ax[1].set_title("att map")
                cb4 = logit_fig.colorbar(
                    im2, location="right", shrink=0.99, pad=0.02, ax=ax[1]
                )
                ax[1].set_xticks(range(avg_attn_map.shape[-1]))
                ax[1].set_yticks(range(avg_attn_map.shape[-2]))

                log_data.update(
                    {
                        "logit_fig": logit_fig,
                    }
                )

            wandb.log(log_data)

            if step % 10 == 0:
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
                        "test_loss": test_loss,
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

    config.model.vocab_size = max(config.data.p, config.data.max_num) + 1
    config.model.block_size = 2 * config.data.num_tokens + 1

    data_sampler = MixedMovingWindowSum(
        min_num=config.data.min_num,
        max_num=config.data.max_num,
        k=config.data.k,
        p=config.data.p,
        mix_type=config.data.mix_type,
    )

    model = GPTLinear(config.model, return_att=True).to(device)
    optim = Adam(model.parameters(), lr=config.train.lr)

    if config.train.wandb:
        wandb_run_name = ckpt_dir
        wandb.login(key="")
        wandb.init(
            project="tf-emergence",
            name=f"mws_mix_{config.data.mix_type}",
            config=config,
        )
        wandb.watch(model)

    for step in range(config.train.num_steps):
        train_step(
            model=model,
            optim=optim,
            data_sampler=data_sampler,
            step=step,
            config=config,
        )

