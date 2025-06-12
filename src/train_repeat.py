import yaml
import argparse
from dotmap import DotMap

import torch
from torch.optim import Adam
from torch.nn.functional import cosine_similarity

import wandb
import matplotlib.pyplot as plt

from model_linear import GPTLinear
from data import Repeat

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
    _, _, _, loss = model(
        train_data[:, :-1], targets=train_data[:, 1:]
    )
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    model.eval()
    with torch.no_grad():
        # log train loss, train and test acc
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


        # Log cosine similarity 
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

        if config.data.k == 1:
            f1_approx_acc = torch.mean( 
                (train_pred[:, acc_start:] == train_pred[:, acc_start].view(-1, 1).repeat(1, num_tokens)).to(float)
            ).item()
            mean_cosine_sim = torch.sum(logit_cs[:, 1:]) / (0.5 * (gen_len-1) * (gen_len-2))

        elif config.data.k == 2:
            first_half_acc = torch.mean(
                (train_pred[:, acc_start: acc_start+num_tokens//2] == train_data[:, acc_start: acc_start+num_tokens//2]).to(float)
            ).item()

            second_half_acc = torch.mean( 
                (train_pred[:, acc_start+num_tokens//2:] == train_data[:, acc_start+num_tokens//2:]).to(float)
            ).item()

            f1_approx_acc = torch.mean( 
                (train_pred[:, acc_start:] == train_pred[:, acc_start].view(-1, 1).repeat(1, num_tokens)).to(float)
            ).item()
            mean_cosine_sim = (torch.sum(logit_cs[2:8, 1:7]) + torch.sum(logit_cs[10:16, 9:15])) / 42


        elif config.data.k == 4:
            first_quart_acc = torch.mean(
                (train_pred[:, acc_start: acc_start+num_tokens//4] == train_data[:, acc_start: acc_start+num_tokens//4]).to(float)
            ).item()
            second_quart_acc = torch.mean(
                (train_pred[:, acc_start+num_tokens//4: acc_start+num_tokens//2] == train_data[:, acc_start+num_tokens//4: acc_start+num_tokens//2]).to(float)
            ).item()
            third_quart_acc = torch.mean(
                (train_pred[:, acc_start+num_tokens//2: acc_start+(3*num_tokens)//4] == train_data[:, acc_start+num_tokens//2: acc_start+(3*num_tokens)//4]).to(float)
            ).item()
            fourth_quart_acc = torch.mean(
                (train_pred[:, acc_start+(3*num_tokens)//4:] == train_data[:, acc_start+(3*num_tokens)//4:]).to(float)
            ).item()
            
            f1_approx_acc = torch.mean( 
                (train_pred[:, acc_start:] == train_pred[:, acc_start].view(-1, 1).repeat(1, num_tokens)).to(float)
            ).item()

            f2_approx_acc_1 = torch.mean( 
                (
                    (train_pred[:, acc_start: acc_start+num_tokens//2] == train_pred[:, acc_start].view(-1, 1).repeat(1, num_tokens//2)).to(float)
                )
            ).item()
            
            f2_approx_acc_2 = torch.mean( 
                (
                    (train_pred[:, acc_start+num_tokens//2:] == train_pred[:, acc_start+num_tokens//2].view(-1, 1).repeat(1, num_tokens//2)).to(float)
                )
            ).item()

            mean_cosine_sim = (
                logit_cs[2, 1] + logit_cs[3, 1] + logit_cs[3, 2] +
                logit_cs[6, 5] + logit_cs[7, 5] + logit_cs[7, 6] +
                logit_cs[10, 9] + logit_cs[11, 9] + logit_cs[11, 10] +
                logit_cs[14, 13] + logit_cs[15, 13] + logit_cs[15, 14]
            ) / 12

        
        
        logit_fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))
        
        im1 = ax[0].imshow(logit_cs)
        ax[0].set_title("avg pre_lm_h cosine sim")
        cb1 = logit_fig.colorbar(im1, location="right", shrink=0.99, pad=0.02, ax=ax[0])

        avg_attn_map = torch.mean(attn_map, dim=0).squeeze().detach().cpu().numpy()

        im4 = ax[1].imshow(avg_attn_map)
        ax[1].set_title("att map")
        cb4 = logit_fig.colorbar(im4, location="right", shrink=0.99, pad=0.02, ax=ax[1])
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
                "pre_lm_h cosine sim": logit_fig,
                "data_repeat_frac": data_repeat_frac,
                "model_repeat_frac": model_repeat_frac,
                "mean_cosine_sim": mean_cosine_sim
            }   

            for output_pos in range(gen_len-1):
                log_data.update(
                    {
                        f"mean_cosine_sim_{output_pos}": torch.sum(logit_cs[:, output_pos]) / (gen_len-1-output_pos)
                    }
                )
            
            if config.data.k == 1:
                log_data.update({
                    'f1_approx_acc': f1_approx_acc
                })

            if config.data.k == 2:
                log_data.update({
                    'first_half_acc': first_half_acc,
                    'second_half_acc': second_half_acc,
                    'f1_approx_acc': f1_approx_acc
                })

            elif config.data.k == 4:
                log_data.update({
                    'first_quart_acc': first_quart_acc,
                    'second_quart_acc': second_quart_acc,
                    'third_quart_acc': third_quart_acc,
                    'fourth_quart_acc': fourth_quart_acc,
                    'f1_approx_acc': f1_approx_acc,
                    'f2_approx_acc_1': f2_approx_acc_1,
                    'f2_approx_acc_2': f2_approx_acc_2,
                })

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
    
    config.model.vocab_size = config.data.max_num + 2
    config.model.block_size = 2 * config.data.num_tokens + 1

    data_sampler = Repeat(
        min_num=config.data.min_num,
        max_num=config.data.max_num,
        pos=config.data.pos,
        k=config.data.k,
    )

    model = GPTLinear(config.model, return_att=True).to(device)
    optim = Adam(model.parameters(), lr=config.train.lr)  

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

