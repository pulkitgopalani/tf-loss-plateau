import os

import numpy as np

import torch
from torch.nn.functional import cosine_similarity

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import matplotlib.pyplot as plt

revision = {
    0: "step0",
    1: "step1",
    2: "step2",
    4: "step4",
    8: "step8",
    16: "step16",
    32: "step32",
    64: "step64",
    128: "step128",
    256: "step256",
    512: "step512",
    1000: "step1000",
    2000: "step2000",
    4000: "step4000",
    8000: "step8000",
    10000: "step10000",
}

# 14m, 1b, 1.4b, 2.8b
model_version = "14m"

org_name = "allenai/ai2_arc"
dataset_name = "ARC-Easy"
num_questions = 100


questions = (
    load_dataset(org_name, name=dataset_name, split="test")
    .shuffle()
    .select(range(num_questions))["question"]
)

model_str = f"EleutherAI/pythia-{model_version}"

num_gen_tokens = 8
do_sample = False

avg_cosine_sims = []

model_result_dir = f"./pythia_final{'_sample' if do_sample else ''}/arc-easy/model_{model_version}"
if not os.path.exists(model_result_dir):
    os.mkdir(model_result_dir)

all_repetitions = torch.zeros(
    (
        len(revision.keys()),
        num_questions,
    )
)

for rev_idx, k in enumerate(revision.keys()):
    pythia = AutoModelForCausalLM.from_pretrained(model_str, revision=revision[k])
    tokenizer = AutoTokenizer.from_pretrained(model_str, revision=revision[k])

    logit_cs = torch.zeros((len(questions), num_gen_tokens, num_gen_tokens))

    print(f"step {k} model and tokenizer loaded")

    # repetitions = []
    for ques_idx, question in enumerate(questions):
        inputs = tokenizer(question, return_tensors="pt", return_token_type_ids=False)

        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        pythia = pythia.to("cuda")

        model_output = pythia.generate(
            **inputs,
            min_new_tokens=num_gen_tokens,
            max_new_tokens=num_gen_tokens,
            do_sample=do_sample,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=False,
        )

        response = model_output.sequences

        all_repetitions[rev_idx, ques_idx] = torch.mean(
            (response[0, -num_gen_tokens:-1] == response[0, -num_gen_tokens + 1 :]).to(
                float
            )
        ).item()

        all_hidden_states = model_output.hidden_states

        for i_1 in range(num_gen_tokens):
            for i_2 in range(i_1):
                logit_cs[ques_idx, i_1, i_2] = cosine_similarity(
                    all_hidden_states[i_1][-1][:, -1, :].squeeze(),
                    all_hidden_states[i_2][-1][:, -1, :].squeeze(),
                    dim=-1,
                )

                # logit_cs[ques_idx, i_1, i_2] = cosine_similarity(
                #     all_hidden_states[i_1][-1][:, -1, :].squeeze(),
                #     all_hidden_states[i_2][-1][:, -1, :].squeeze(),
                #     dim=-1,
                # )

    logit_cs_np = logit_cs.detach().cpu().numpy()
    np.save(f"{model_result_dir}/cs_step_{k}.npy", logit_cs_np)


    avg_logit_cs = torch.mean(logit_cs, dim=0)
    avg_cosine = torch.sum(avg_logit_cs) / (0.5 * num_gen_tokens * (num_gen_tokens - 1))
    print(f"step {k} avg cosine: {avg_cosine}")
    avg_cosine_sims.append(avg_cosine)

    logit_fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    im1 = ax.imshow(avg_logit_cs)
    ax.set_title("avg post-mlp cosine sim")
    cb1 = logit_fig.colorbar(im1, location="right", shrink=0.99, pad=0.02, ax=ax)

    for i1 in range(num_gen_tokens):
        for i2 in range(num_gen_tokens):
            text1 = ax.text(
                i2,
                i1,
                round(avg_logit_cs[i1, i2].item(), 2),
                ha="center",
                va="center",
                color="w",
            )

    plt.savefig(os.path.join(model_result_dir, f"step_{k}.png"))
    plt.close()

repetition_np = all_repetitions.cpu().numpy()
np.save(f"{model_result_dir}/rep_step_{k}.npy", repetition_np)

print(
    f"{model_version} | ",
    " | ".join(
        [str(round(rep, 2)) for rep in torch.mean(all_repetitions, dim=-1).tolist()]
    ),
    " | ",
)

print(
    f"{dataset_name}_{model_version} | ",
    " | ".join([str(round(cs.item(), 3)) for cs in avg_cosine_sims]),
    " | ",
)

plt.plot(list(revision.keys()), avg_cosine_sims)
plt.xscale("log")
plt.savefig(os.path.join(model_result_dir, "cosine_sim_evolution.png"))
plt.close()


plt.plot(torch.mean(all_repetitions, dim=-1))
plt.savefig(os.path.join(model_result_dir, "repetitions.png"))
plt.close()
