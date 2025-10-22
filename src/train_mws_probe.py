import yaml
import argparse
from dotmap import DotMap

import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity

from sklearn.neural_network import MLPClassifier


import wandb
import matplotlib.pyplot as plt

from model_linear_probe import GPTLinear
from data import MovingWindowSum

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(threshold=100000000)


def get_mlp_probe_acc(hidden_states, targets, d_probe):

    n = hidden_states.shape[0]
    n_train = hidden_states.shape[0] // 2
    
    L = hidden_states.shape[1]
    d = hidden_states.shape[2]

    hidden_states = hidden_states.reshape(n * L, d)
    targets = targets.reshape(
        n * L,
    )

    X_train = hidden_states[:n_train]
    y_train = targets[:n_train]

    X_test = hidden_states[n_train:]
    y_test = targets[n_train:]

    classifier = MLPClassifier(hidden_layer_sizes=(d_probe,), max_iter=2000).fit(
        X_train, y_train
    )

    train_acc = classifier.score(X_train, y_train)

    # print(f"Probe train acc: {train_acc}")
    test_acc = classifier.score(X_test, y_test)

    return test_acc, train_acc


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

    _, _, _, loss = model(train_data[:, :-1], targets=train_data[:, 1:])
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    model.eval()
    with torch.no_grad():
        # Log train loss, train / test acc, repetition frequency
        attn_map, post_att_pre_res_h, _, test_loss = model(
            test_data[:, :-1], targets=test_data[:, 1:]
        )

        _, _, _, train_loss = model(
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
                            post_att_pre_res_h[:, i_1, :],
                            post_att_pre_res_h[:, i_2, :],
                            dim=-1,
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

            avg_attn_map = torch.mean(torch.abs(attn_map), dim=0).squeeze().detach().cpu().numpy()

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
        print(f"Step {step} -- Test loss: {test_loss}, Test Acc: {test_acc}")
        # print(f"input: {test_data[0]} \n predicted:{test_pred[0]}")

        if config.train.wandb:

            log_data = {
                "test_loss": test_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "model_repeat_frac": model_repeat_frac,
                "att_prog_measure": att_prog_measure,
                "mean_cosine_sim": torch.sum(logit_cs[:, 1:])
                / (0.5 * (gen_len - 1) * (gen_len - 2)),
            }

            if step % config.train.probe_freq == 0:
                probe_test_acc, probe_train_acc = get_mlp_probe_acc(
                    post_att_pre_res_h.clone().detach().cpu().numpy()[:, num_tokens:],
                    test_data.clone().detach().cpu().numpy()[:, num_tokens + 1 :],
                    config.model.d_probe,
                )
                log_data.update(
                    {
                        "probe_test_acc": probe_test_acc,
                        "probe_train_acc": probe_train_acc,
                    }
                )

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

    data_sampler = MovingWindowSum(
        min_num=config.data.min_num,
        max_num=config.data.max_num,
        k=config.data.k,
        p=config.data.p,
    )

    model = GPTLinear(config.model, return_att=True).to(device)
    optim = Adam(model.parameters(), lr=config.train.lr)

    if config.train.wandb:
        wandb_run_name = ckpt_dir
        wandb.login(key="")
        wandb.init(
            project="tf-emergence",
            name=f"mws_probe_relu_p{config.model.d_probe}_d{4*config.model.n_embd}_n{config.data.n_test // 2}",
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
