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
    parser.add_argument("--data")
    
    data = parser.parse_args().data

    do_sample = False

    fig_name = f"./pythia_{data}"
    model_name_to_dir = {
        '14m': f"./pythia{'_sample' if do_sample else ''}_final/{data}/model_14m",
        '1b': f"./pythia{'_sample' if do_sample else ''}_final/{data}/model_1b",
        '1.4b': f"./pythia{'_sample' if do_sample else ''}_final/{data}/model_1.4b",
        '2.8b': f"./pythia{'_sample' if do_sample else ''}_final/{data}/model_2.8b",
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True)

    colors = cm.get_cmap('viridis', 4)
    steps = list(revision.keys())

    for model_idx, model_name in enumerate(model_name_to_dir.keys()):
        # logit_cs_dict = {}
        mean_cosine_sim = []
        for step_idx, step in enumerate(steps):
            logit_cs = np.load(f'{model_name_to_dir[model_name]}/cs_step_{step}.npy')
            mean_cosine_sim.append(np.mean(np.sum(logit_cs, axis=(-2, -1))/(0.5 * gen_len * (gen_len-1)), axis=0))
        
        repetitions = np.load(f'{model_name_to_dir[model_name]}/repetition.npy')
        mean_repetition = np.mean(repetitions, axis=-1)

        ax_left = axes[model_idx]
        ax_right = ax_left.twinx()

        ax_left.plot(steps, mean_cosine_sim, color=colors(0), linewidth=2.5, markersize=5, label='Cosine Sim.')
        ax_left.scatter(steps, mean_cosine_sim, color=colors(0), linewidth=2.5)

        ax_right.plot(steps, mean_repetition, color='red', linewidth=2.5, markersize=5, linestyle='--', label='Repeat Freq.')
        ax_right.scatter(steps, mean_repetition, color='red', linewidth=2.5, marker='^')

        ax_left.set_title(f'Pythia-{model_name}', fontsize=18)
        ax_left.set_xlabel('Pretraining Step', fontsize=16)
        if model_idx == 0:
            ax_left.set_ylabel('Cosine Similarity', fontsize=16)
        if model_idx == 3:
            ax_right.set_ylabel('Repetition Frequency', fontsize=16)
            lines1, labels1 = ax_left.get_legend_handles_labels()
            lines2, labels2 = ax_right.get_legend_handles_labels()
            ax_left.legend(lines1+lines2, labels1+labels2, loc='upper right', frameon=True, shadow=True)
            
        ax_left.set_xscale('log')
            

        ax_left.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
        ax_left.tick_params(axis='x', labelsize=14)
        ax_left.tick_params(axis='y', labelsize=14)
        ax_right.tick_params(axis='y', labelsize=14)

        
        ax_left.set_facecolor('#f9f9f9')

    fig.tight_layout(w_pad=1.0)

    plt.savefig(f'./figs/{fig_name}.png', format='png')
    plt.savefig(f'./figs/{fig_name}.pdf', format='pdf')