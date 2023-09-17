from cmx import doc

if __name__ == '__main__':

    doc @ """
    # Learning Speed-up




    """
    import os
    import matplotlib.pyplot as plt
    from cmx import doc
    from ml_logger import ML_Logger, memoize
    from tqdm import tqdm

    colors = ['#23aaff', '#ff7575', '#66c56c', '#f4b247', 'orange', 'magenta', 'brown']

    with doc:
        loader = ML_Logger(prefix="/geyang/scratch/2022/06-17/mit/noisy-ntk/dnp_analysis/"
                                  "proof_of_concept/sine_regression/00.08.38-finetune/1")
    loader.glob = memoize(loader.glob)
    loader.read_metrics = memoize(loader.read_metrics)

    with doc:
        def plot_line(path, color, label, x_key, y_key, line_style=None):
            mean, low, high, step, = loader.read_metrics(
                f"{y_key}/loss/mean@mean",
                f"{y_key}/loss/mean@16%",
                f"{y_key}/loss/mean@84%",
                x_key=f"{x_key}@min", path=path, dropna=True)
            plt.xlabel('Epochs', fontsize=18)
            plt.ylabel('L1 Loss', fontsize=18)

            plt.plot(step.to_list(), mean.to_list(), color=color, label=label, linestyle=line_style)
            plt.fill_between(step, low, high, alpha=0.1, color=color)

    with doc, doc.table() as t:
        r = t.figure_row()

        for i, scale in enumerate([1, 3, 5, 10, 20, 40, 80]):
            plot_line(path=f"metrics.pkl", color=colors[i], label=f'FFN({scale})', x_key='epoch', y_key=f'ffn-{scale}')

        plt.title("Fourier Features (scale)")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.tight_layout()
        r.savefig(f'{os.path.basename(__file__)[:-3]}/ffn_all.png', dpi=300, zoom=0.3,
                  title=f"Fourier Features (scale)")
        plt.savefig(f'{os.path.basename(__file__)[:-3]}/ffn_all.pdf')
        plt.close()

        for i, epoch in enumerate(range(0, 1200_000, 200_000)):
            plot_line(path=f"metrics.pkl", color=colors[i], label=f'MLP {epoch // 1000}k', x_key='epoch',
                      y_key=f'mlp-{epoch // 1000}k')

        plt.title("Pre-training (epoch)")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.tight_layout()
        r.savefig(f'{os.path.basename(__file__)[:-3]}/mlp_all.png', dpi=300, zoom=0.3, title=f"Pre-training")
        # plt.savefig(f'{os.path.basename(__file__)[:-3]}/mlp_all.pdf')
        plt.close()

        r = t.figure_row()
        epoch = 0
        plot_line(path=f"metrics.pkl", color="black", line_style="--",
                  label=f'MLP (vanilla)', x_key='epoch',
                  y_key=f'mlp-{epoch // 1000}k')
        epoch = 200_000
        plot_line(path=f"metrics.pkl", color="orange", label=f'MLP@{epoch // 1000}k', x_key='epoch',
                  y_key=f'mlp-{epoch // 1000}k')
        epoch = 1000_000
        plot_line(path=f"metrics.pkl", color=colors[0], label=f'MLP@{epoch // 1000}k', x_key='epoch',
                  y_key=f'mlp-{epoch // 1000}k')
        scale = 40
        plot_line(path=f"metrics.pkl", color=colors[1], label=f'FFN(b={scale})', x_key='epoch', y_key=f'ffn-{scale}')
        plt.title("Linear Regression")
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.tight_layout()
        r.savefig(f'{os.path.basename(__file__)[:-3]}/full_result.png', dpi=300, zoom=0.3, title=f"Full Result")
        # plt.savefig(f'{os.path.basename(__file__)[:-3]}/mlp_{scale}.pdf')
        plt.close()

    doc.flush()
