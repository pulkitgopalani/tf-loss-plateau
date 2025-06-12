import yaml
import argparse
from dotmap import DotMap

import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity

import wandb
import matplotlib.pyplot as plt

from model_linear import GPTLinear
from data import MovingWindowSum

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
    
    seq_len = num_tokens
    prompt_len = seq_len + 1
    gen_len = seq_len
    acc_start = seq_len + 1

    model.train()
    optim.zero_grad(set_to_none=True)

    attn_map, post_mlp_h, _, loss = model(
        train_data[:, :-1], targets=train_data[:, 1:]
    )
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()
    
    model.eval()

    with torch.no_grad():
        # Log train loss, train / test acc, repetition frequency
        # train_loss = loss.clone().detach().item()
        
        _, _, _, train_loss = model(train_data[:, :-1], targets=train_data[:, 1:])
        _, _, _, test_loss = model(test_data[:, :-1], targets=test_data[:, 1:])
        
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
        attn_map_output_seq = attn_map[:, :, num_tokens:]
        att_mask = torch.zeros_like(attn_map_output_seq).to(device)

        if config.data.k == 2:
            att_mask[:, :, 0, 0] = 1
            for i in range(num_tokens - 1):
                att_mask[:, :, i + 1, i : i + 2] = 1
        elif config.data.k == 3:
            att_mask[:, :, 0, 0] = 1
            att_mask[:, :, 1, 1] = 1
            for i in range(num_tokens - 2):
                att_mask[:, :, i + 2, i : i + 2] = 1

        att_prog_measure = torch.sum(torch.abs(attn_map_output_seq) * att_mask) / torch.sum(
            torch.abs(attn_map_output_seq)
        )

        
        # Log pair-wise cosine similarity between hidden states
        embed_start = config.data.num_tokens
        embed_len = config.data.num_tokens

        logit_cs = torch.zeros((embed_len, embed_len))
        
        for i_1 in range(embed_start, embed_start + embed_len):
            for i_2 in range(embed_start, i_1):
                logit_cs[i_1 - embed_start, i_2 - embed_start] = torch.mean(
                    (
                        cosine_similarity(
                            post_mlp_h[:, i_1, :], post_mlp_h[:, i_2, :], dim=-1
                        )
                    ), dim=0
                )
                
        # Log plots for cosine similarity, attention map
        logit_fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

        im1 = ax[0].imshow(logit_cs)
        ax[0].set_title("avg post-mlp cosine sim")
        cb1 = logit_fig.colorbar(im1, location="right", shrink=0.99, pad=0.02, ax=ax[0])

        avg_attn_map = torch.mean(attn_map, dim=0).squeeze().detach().cpu().numpy()
        
        im2 = ax[1].imshow(avg_attn_map)
        ax[1].set_title("att map")
        cb4 = logit_fig.colorbar(im2, location="right", shrink=0.99, pad=0.02, ax=ax[1])
        ax[1].set_xticks(range(attn_map.shape[-1]))
        ax[1].set_yticks(range(attn_map.shape[-2]))

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
        

        idx_check = []
        for output_pos in range(num_tokens):
            idx_check.append(torch.mean(
                    (train_pred[:, acc_start + output_pos] == train_data[:, acc_start + output_pos]).to(float)
                ).item()
            )
            
        print(
            f"Step {step} -- Train loss: {train_loss}, Train Acc: {train_acc} Test Acc: {test_acc}"
        )
        print(f"input: {test_data[0]} \n predicted:{test_pred[0]}")

        if config.train.wandb:

            log_data = {
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "data_repeat_frac": data_repeat_frac,
                "model_repeat_frac": model_repeat_frac,
                "att_prog_measure": att_prog_measure,
                "post_mlp_h_cosine_sim": logit_fig,
                "mean_cosine_sim": torch.sum(logit_cs[:, config.data.k-1:]) / (0.5 * (num_tokens-1) * (num_tokens-2))
            }

            for output_pos in range(num_tokens):
                log_data.update({f"idx{output_pos}_check": idx_check[output_pos]})

                if output_pos < num_tokens-1:
                    log_data.update(
                        {
                            f"mean_cosine_sim_{output_pos}": torch.sum(logit_cs[:, output_pos]) / (num_tokens-1-output_pos)
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

    # Freeze any layer in the model at random initialization
    for name, param in model.named_parameters():
            if (
                # ("attn" in name)
                ("mlp" in name)
                # or ("wte" in name)
                # or ("wpe" in name)
                # or ("ln_1" in name)
                # or ("ln_2" in name)
                # or ("ln_f" in name)
                # or ("lm_head" in name)
            ):
                param.requires_grad = False
                print(f"Frozen {name} at init")

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

    