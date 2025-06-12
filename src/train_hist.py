from dotmap import DotMap
import yaml
import argparse

import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity

import wandb
import matplotlib.pyplot as plt

from model_linear_2layer import GPTLinear
from data import Histogram

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
    gen_len = num_tokens
    acc_start = num_tokens + 1

    model.train()

    optim.zero_grad(set_to_none=True)
    _, _, _, _, _, _, loss = model(
        train_data[:, :-1], targets=train_data[:, 1:]
    )
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    model.eval()
    with torch.no_grad():
        # Log train loss, train / test acc, count check, repetition frequency
        attn_map1, attn_map2, pre_lm_h, post_mlp_h_1, post_mlp_h_2, _, train_loss = model(
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

        data_repeat_frac = torch.mean((test_data[:, acc_start:-1] == test_data[:, acc_start+1:]).to(float))
        model_repeat_frac = torch.mean((test_pred[:, acc_start:-1] == test_pred[:, acc_start+1:]).to(float))

        # Log attention progress measure 
        attn_map_output_seq1 = attn_map1[:, :, acc_start-1:, :]
        att_mask1 = torch.zeros_like(attn_map_output_seq1).to(device)

        for i in range(num_tokens):
            att_mask1[:, :, i, i] = 1
    
        
        att_prog_measure1 = torch.mean(
            torch.sum(torch.abs(attn_map_output_seq1) * att_mask1, dim=(-3, -2, -1)) / 
            torch.sum(torch.abs(attn_map_output_seq1), dim=(-3, -2, -1)),
            dim=0
        )

        # Log pair-wise cosine similarity between hidden states
        embed_start = acc_start - 1
        embed_len = gen_len
        
        post_mlp_cs_1 = torch.zeros((embed_len, embed_len))
        post_mlp_cs_2 = torch.zeros((embed_len, embed_len))
        pre_lm_cs = torch.zeros((embed_len, embed_len))

        for i_1 in range(embed_start, embed_start+embed_len):
            for i_2 in range(embed_start, i_1):
                post_mlp_cs_1[i_1 - embed_start, i_2 - embed_start] = torch.mean(
                    (
                        cosine_similarity(
                            post_mlp_h_1[:, i_1, :], post_mlp_h_1[:, i_2, :], dim=-1
                        )
                    ),
                    dim=0,
                )

                post_mlp_cs_2[i_1 - embed_start, i_2 - embed_start] = torch.mean(
                    (
                        cosine_similarity(
                            post_mlp_h_2[:, i_1, :], post_mlp_h_2[:, i_2, :], dim=-1
                        )
                    ),
                    dim=0,
                )

                pre_lm_cs[i_1 - embed_start, i_2 - embed_start] = torch.mean(
                    (
                        cosine_similarity(
                            pre_lm_h[:, i_1, :], pre_lm_h[:, i_2, :], dim=-1
                        )
                    ),
                    dim=0,
                )
        
        if step % 10 == 0:
            # Log plots for cosine similarity, attention map
            logit_fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(75, 60))
            
            im1 = ax[0, 0].imshow(post_mlp_cs_1)
            ax[0, 0].set_title("avg post-mlp cosine sim 1")
            cb1 = logit_fig.colorbar(im1, location="right", shrink=0.99, pad=0.02, ax=ax[0, 0])
            
            im2 = ax[0, 1].imshow(post_mlp_cs_2)
            ax[0, 1].set_title("avg post-mlp cosine sim 2")
            cb2 = logit_fig.colorbar(im2, location="right", shrink=0.99, pad=0.02, ax=ax[0, 1])

            im5 = ax[0, 2].imshow(pre_lm_cs)
            ax[0, 2].set_title("avg pre-lm cosine sim")
            cb5 = logit_fig.colorbar(im5, location="right", shrink=0.99, pad=0.02, ax=ax[0, 2])
            
            random_idx = torch.randint(low=0, high=config.data.n_train, size=(1,)).item()

            attn_map1 = (attn_map1[random_idx].squeeze().detach().cpu().numpy())
            attn_map2 = (attn_map2[random_idx].squeeze().detach().cpu().numpy())

            im3 = ax[1, 0].imshow(attn_map1)
            ax[1, 0].set_title("att map 1")
            cb3 = logit_fig.colorbar(im3, location="right", shrink=0.99, pad=0.02, ax=ax[1, 0])

            im4 = ax[1, 1].imshow(attn_map2)
            ax[1, 1].set_title("att map 2")
            cb4 = logit_fig.colorbar(im4, location="right", shrink=0.99, pad=0.02, ax=ax[1, 1])

            plt.suptitle(f"{train_data[random_idx].detach().cpu().tolist()}\n{train_pred[random_idx].detach().cpu().tolist()}", fontsize=15)

            for i1 in range(attn_map1.shape[0]):
                for i2 in range(attn_map1.shape[1]):
                    text1 = ax[1, 0].text(
                        i2,
                        i1,
                        round(attn_map1[i1, i2].item(), 2),
                        ha="center",
                        va="center",
                        color="w",
                    )
                    text2 = ax[1, 1].text(
                        i2,
                        i1,
                        round(attn_map2[i1, i2].item(), 2),
                        ha="center",
                        va="center",
                        color="w",
                    )

            for i1 in range(post_mlp_cs_1.shape[0]):
                for i2 in range(post_mlp_cs_1.shape[1]):
                    text3 = ax[0, 0].text(
                        i2,
                        i1,
                        round(post_mlp_cs_1[i1, i2].item(), 2),
                        ha="center",
                        va="center",
                        color="w",
                    )
                    text4 = ax[0, 1].text(
                        i2,
                        i1,
                        round(post_mlp_cs_2[i1, i2].item(), 2),
                        ha="center",
                        va="center",
                        color="w",
                    )
                    text5 = ax[0, 2].text(
                        i2,
                        i1,
                        round(pre_lm_cs[i1, i2].item(), 2),
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
                "train acc": train_acc,
                "test_acc": test_acc,
                "att_prog_measure1": att_prog_measure1,
                "data_repeat_frac": data_repeat_frac,
                "model_repeat_frac": model_repeat_frac,
            }

            for output_pos in range(gen_len - 1):
                log_data.update(
                    {
                        f"idx{output_pos}_check": torch.mean(
                            (train_pred[:, acc_start + output_pos] == train_data[:, acc_start + output_pos]).to(float)
                        ).item()
                    }
                )

                if output_pos < gen_len - 1:
                    log_data.update(
                        {
                            f"mean_cosine_sim_1_{output_pos}": torch.sum(post_mlp_cs_1[:, output_pos]) / (embed_len - output_pos - 1),
                            f"mean_cosine_sim_2_{output_pos}": torch.sum(post_mlp_cs_2[:, output_pos]) / (embed_len - output_pos - 1),
                            f"mean_cosine_pre_lm_{output_pos}": torch.sum(pre_lm_cs[:, output_pos]) / (embed_len - output_pos - 1)
                        }
                    )

            log_data.update(
                {
                    f"mean_cosine_sim_1": torch.sum(post_mlp_cs_1) / (0.5 * gen_len * (gen_len-1)),
                    f"mean_cosine_sim_2": torch.sum(post_mlp_cs_2) / (0.5 * gen_len * (gen_len-1)),
                    f"mean_cosine_pre_lm": torch.sum(pre_lm_cs) / (0.5 * gen_len * (gen_len-1)),
                }
            )

            if step % 10 == 0:
                log_data.update({"att_map": logit_fig})

            wandb.log(log_data)

        if step % 10 == 0:
            plt.close()
            del logit_fig, ax

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

    config.model.vocab_size = max(config.data.alphabet_size + 1, config.data.num_tokens + 1)
    config.model.block_size = 2 * config.data.num_tokens + 1

    data_sampler = Histogram(alphabet_size=config.data.alphabet_size, device=device)

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
