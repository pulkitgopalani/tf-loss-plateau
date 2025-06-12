import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

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

gen_len = 8

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot")
    plot = parser.parse_args().plot

    png = False
    do_sample = True

    fig_name = f"./pythia{'_sample' if do_sample else ''}"
    model_name_to_dir = {
        '14m': f"./pythia{'_sample' if do_sample else ''}/model_14m",
        '1b': f"./pythia{'_sample' if do_sample else ''}/model_1b",
        '1.4b': f"./pythia{'_sample' if do_sample else ''}/model_1.4b",
        '2.8b': f"./pythia{'_sample' if do_sample else ''}/model_2.8b",
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)

    colors = cm.get_cmap('viridis', 4)
    steps = list(revision.keys())

    for model_idx, model_name in enumerate(model_name_to_dir.keys()):
        # logit_cs_dict = {}
        mean_cosine_sim = []
        for step_idx, step in enumerate(steps):
            logit_cs = np.load(f'{model_name_to_dir[model_name]}/results_step_{step}.npy')
            mean_cosine_sim.append(np.mean(np.sum(logit_cs, axis=(-2, -1))/(0.5 * gen_len * (gen_len-1)), axis=0))
        
        axes[model_idx].plot(steps, mean_cosine_sim, color=colors(0), linewidth=2.5, markersize=5)
        axes[model_idx].scatter(steps, mean_cosine_sim, color=colors(0), linewidth=2.5)

        axes[model_idx].set_title(f'Pythia-{model_name}', fontsize=18)
        axes[model_idx].set_xlabel('Pretraining Step', fontsize=16)
        if model_idx == 0:
            axes[model_idx].set_ylabel('Cosine Similarity', fontsize=16)
        axes[model_idx].set_xscale('log')
            

        axes[model_idx].grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
        axes[model_idx].tick_params(axis='x', labelsize=14)
        axes[model_idx].tick_params(axis='y', labelsize=14)

        
        # axes[model_idx].legend(fontsize=14, loc='center right', frameon=True, shadow=True)
        axes[model_idx].set_facecolor('#f9f9f9')




    # fig.suptitle(r'Training Dynamics for $MWS_2$', fontsize=22, weight='bold')
    # fig.subplots_adjust(wspace=)
    fig.tight_layout(w_pad=1.0)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.patch.set_facecolor('#f0f0f0')

    if png:
        plt.savefig(f'./figs/{fig_name}.png')
    
    else:
        plt.savefig(f'./figs/{fig_name}.pdf', format='pdf')































