import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# BET lift
bet_lift, bet_lift_times = np.loadtxt('lift/mini_bet_state.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_lift_scores = [0, 0.35, 0.65, 0.95, 0.95, 0.95, 1]
bet_lift_vision, bet_lift_vision_times = np.loadtxt('lift/mini_bet_vision.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_lift_vision_scores = [0, 0, 0.5, 0.3, 0.85, 0.85, 0.7, 0.85, 0.8, 0.65, 1]
bet_lift_hybrid, bet_lift_hybrid_times = np.loadtxt('lift/mini_bet_hybrid.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_lift_hybrid_scores = [0, 0, 0.4, 0.5, 0.65, 0.65, 0.9, 0.85, 0.9, 0.95, 0.9]

# DP lift
dp_lift, dp_lift_times = np.loadtxt('lift/dp_state_old.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_lift_scores = [0, 0.7, 0.9, 0.9, 0.95, 0.95, 0.9, 0.95, 0.95, 1]
dp_lift_vision, dp_lift_vision_times = np.loadtxt('lift/dp_vision.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_lift_vision_scores = [0, 0.5, 0.75, 0.6, 0.45, 0.65, 1]
dp_lift_hybrid, dp_lift_hybrid_times = np.loadtxt('lift/dp_hybrid.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_lift_hybrid_scores = [0, 0.45, 0.55, 0.3, 0.85, 0.5, 0.65, 1]

# BET can
bet_can, bet_can_times = np.loadtxt('can/mini_bet_state.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_can_scores = [0, 0.95, 0.95, 1]
bet_can_vision, bet_can_vision_times = np.loadtxt('can/mini_bet_vision.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_can_vision_scores = [0, 0.05, 0, 0.1, 0.1, 0.15, 0, 0, 0.05, 0, 0.1]
bet_can_hybrid, bet_can_hybrid_times = np.loadtxt('can/mini_bet_hybrid.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_can_hybrid_scores = [0, 0.15, 0.25, 0.35, 0.25, 0.25, 0.2, 0, 0.05, 0.15, 0.1]

# DP can
dp_can, dp_can_times = np.loadtxt('can/dp_state.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_can_scores = [0, 0.9, 0.85, 1]
dp_can_vision, dp_can_vision_times = np.loadtxt('can/dp_vision.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_can_vision_scores = [0, 0.65, 0.6, 0.8, 0.8, 0.85, 0.55, 0.85, 0.75, 0.65, 0.6]
dp_can_hybrid, dp_can_hybrid_times = np.loadtxt('can/dp_hybrid.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_can_hybrid_scores = [0, 0.35, 0.8, 0.85, 0.75, 0.65, 0.7, 0.85, 0.8, 0.6, 0.9]

# BET square
bet_square, bet_square_times = np.loadtxt('square/mini_bet_state.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_square_scores = [0, 0.45, 0.65, 0.35, 0.35, 0.5, 0.35, 0.35, 0.35, 0.6, 0.7]
bet_square_vision, bet_square_vision_times = np.loadtxt('square/mini_bet_vision.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_square_vision_scores = [0, 0, 0.1, 0.1, 0.15, 0.1, 0.2, 0.2, 0, 0.15, 0.05]
bet_square_hybrid, bet_square_hybrid_times = np.loadtxt('square/mini_bet_hybrid.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_square_hybrid_scores = [0, 0.05, 0.2, 0.35, 0.4, 0.25, 0.25, 0.25, 0.4, 0.25, 0.15]

# DP square
dp_square, dp_square_times = np.loadtxt('square/dp_state.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_square_scores = [0, 0.6, 0.65, 0.8, 0.75, 0.7, 0.6, 0.6, 0.7, 0.55, 0.75]
dp_square_vision = np.loadtxt('square/dp_vision.csv', delimiter=',', usecols=0, skiprows=1)[:2000 * 100]  # dell
dp_square_vision_scores = [0, 0.5, 0.65, 0.65, 0.25, 0.7, 0.5, 0.75, 0.7, 0.8, 0.6]
dp_square_hybrid, dp_square_vision_times = np.loadtxt('square/dp_hybrid.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_square_hybrid_scores = [0, 0.35, 0.6, 0.45, 0.6, 0.85, 0.7, 0.8, 0.7, 0.6, 0.5]

# BET tool hang
bet_toolhang, bet_toolhang_times = np.loadtxt('binning/mini_bet_state_24.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
bet_toolhang_scores = [0, 0, 0.05, 0.1, 0, 0.25, 0.15, 0.1, 0.1, 0.4, 0]

# DP tool hang
dp_toolhang, dp_toolhang_times = np.loadtxt('tool_hang/dp_state.csv', delimiter=',', usecols=(0, 1), skiprows=1, unpack=True)
dp_toolhang_scores = [0, 0.05, 0.3, 0.15, 0.45, 0.2, 0.45, 0, 0.15, 0.3, 0.25]

dp_can_end_effector = [0.6, 0.75, 0.8]  # last 3
# print(np.mean(dp_can_end_effector))


# bet8_loss = np.loadtxt('binning/mini_bet_state_8.csv', delimiter=',', usecols=0, skiprows=1, unpack=True)
# bet16_loss = np.loadtxt('binning/mini_bet_state_8.csv', delimiter=',', usecols=0, skiprows=1, unpack=True)
# bet32_loss = np.loadtxt('binning/mini_bet_state_8.csv', delimiter=',', usecols=0, skiprows=1, unpack=True)
# bet64_loss = np.loadtxt('binning/mini_bet_state_8.csv', delimiter=',', usecols=0, skiprows=1, unpack=True)

bet8 = 0.15
bet16 = 0.4
bet32 = 0.25
bet64 = 0.55

# Last 3 evals
bet8_scores = [0, 0, 0]
bet16_scores = [0.3, 0.25, 0.4]
bet32_scores = [0.2, 0.2, 0.25]
bet64_scores = [0.4, 0.35, 0.4]

# for x in [bet8_scores, bet16_scores, bet32_scores, bet64_scores]:
#     print(np.mean(x))


def plot_bins():
    fig = plt.figure(figsize=(5, 4))
    x = np.arange(4)
    scores = np.array([bet8_scores, bet16_scores, bet32_scores, bet64_scores])

    plt.bar(x, [bet8, bet16, bet32, bet64], 0.2, color='lightsteelblue', label='max')
    plt.bar(x, np.mean(scores, axis=1), 0.2, color='steelblue', label='mean')
    plt.title('Number of bins comparison')
    plt.xticks(x, [8, 16, 32, 64])
    plt.ylabel('Success Rate')

    plt.legend()
    plt.savefig('bins.pdf')
    plt.show()


# plot_bins()


def print_times():
    bet_times = [bet_lift_times, bet_can_times, bet_square_times, bet_toolhang_times]
    dp_times = [dp_lift_times, dp_can_times, bet_square_times, bet_toolhang_times]
    bet_vision_times = [bet_lift_vision_times, bet_can_vision_times, bet_square_vision_times]
    dp_vision_times = [dp_lift_vision_times, dp_can_vision_times] # dp_square_vision_times]
    bet_hybrid_times = [bet_lift_hybrid_times, bet_can_hybrid_times, bet_square_hybrid_times]
    dp_hybrid_times = [dp_lift_hybrid_times, dp_can_hybrid_times, bet_square_hybrid_times]

    print(np.mean([np.mean(t) * 100 for t in bet_times]))
    print(np.mean([np.mean(t) * 100 for t in dp_times]))
    print(np.mean([np.mean(t) * 100 for t in bet_vision_times]))
    print(np.mean([np.mean(t) * 100 for t in dp_vision_times]))
    print(np.mean([np.mean(t) * 100 for t in bet_hybrid_times]))
    print(np.mean([np.mean(t) * 100 for t in dp_hybrid_times]))


def print_mean(last_n=3):
    scores = [
        # bet_lift_scores, bet_lift_vision_scores, dp_lift_scores, dp_lift_vision_scores,
        # bet_can_scores, bet_can_vision_scores, dp_can_scores, dp_can_vision_scores,
        # bet_square_scores, bet_square_vision_scores, dp_square_scores, dp_square_vision_scores,
        # bet_toolhang_scores, dp_toolhang_scores
        bet_lift_hybrid_scores, bet_can_hybrid_scores, bet_square_hybrid_scores
    ]
    for score in scores:
        print(round(np.mean(score[-last_n:]), 2))


def plot(
    bet_loss, bet_scores, dp_loss, dp_scores,
    bet_loss_vision=None, bet_scores_vision=None, dp_loss_vision=None, dp_scores_vision=None,
    bet_loss_hybrid=None, bet_scores_hybrid=None, dp_loss_hybrid=None, dp_scores_hybrid=None
):
    def x200(x, pos):
        return f'{x * 10:.0f}'

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0][0].plot(bet_loss[100:6100:100], label='state')
    if bet_loss_vision is not None:
        ax[0][0].plot(bet_loss_vision[100:10100:100], label='image')
    if bet_loss_hybrid is not None:
        ax[0][0].plot(bet_loss_hybrid[100::100], label='hybrid')
    ax[0][0].set_title('miniBET')
    ax[0][0].set_xlabel('Epoch')
    ax[0][0].set_ylabel('Loss')
    ax[0][0].legend()

    ax[0][1].plot(bet_scores, linestyle='--')
    ax[0][1].scatter(np.arange(len(bet_scores)), bet_scores, label='state', marker='x', color='#1a5c8d', linewidths=2)
    if bet_scores_vision is not None:
        ax[0][1].plot(bet_scores_vision, linestyle='--')
        ax[0][1].scatter(np.arange(len(bet_scores_vision)), bet_scores_vision, label='image', marker='x', color='#cc6600', linewidths=2)
    if bet_scores_hybrid is not None:
        ax[0][1].plot(bet_scores_hybrid, linestyle='--')
        ax[0][1].scatter(np.arange(len(bet_scores_hybrid)), bet_scores_hybrid, label='hybrid', marker='x', color='#cc6600', linewidths=2)
    ax[0][1].grid(axis='y')
    ax[0][1].set_title('miniBET')
    ax[0][1].set_xlabel('Epoch')
    ax[0][1].set_ylabel('Success Rate')
    ax[0][1].xaxis.set_major_formatter(FuncFormatter(x200))
    ax[0][1].legend()

    ax[1][0].plot(dp_loss[100:9100:100], label='state')
    if dp_loss_vision is not None:
        ax[1][0].plot(dp_loss_vision[100:6100:100], label='image')
    if dp_loss_hybrid is not None:
        ax[1][0].plot(dp_loss_hybrid[100::100], label='hybrid')
    ax[1][0].set_title('Diffusion Policy')
    ax[1][0].set_xlabel('Epoch')
    ax[1][0].set_ylabel('Loss')
    ax[1][0].legend()

    ax[1][1].plot(dp_scores, linestyle='--')
    ax[1][1].scatter(np.arange(len(dp_scores)), dp_scores, label='state', marker='x', color='#1a5c8d', linewidths=2)
    if dp_scores_vision is not None:
        ax[1][1].plot(dp_scores_vision, linestyle='--')
        ax[1][1].scatter(np.arange(len(dp_scores_vision)), dp_scores_vision, label='image', marker='x', color='#cc6600', linewidths=2)
    if dp_scores_hybrid is not None:
        ax[1][1].plot(dp_scores_hybrid, linestyle='--')
        ax[1][1].scatter(np.arange(len(dp_scores_hybrid)), dp_scores_hybrid, label='hybrid', marker='x', color='#cc6600', linewidths=2)
    ax[1][1].grid(axis='y')
    ax[1][1].set_title('Diffusion Policy')
    ax[1][1].set_xlabel('Epoch')
    ax[1][1].set_ylabel('Success Rate')
    ax[1][1].xaxis.set_major_formatter(FuncFormatter(x200))
    ax[1][1].legend()

    plt.tight_layout()
    plt.savefig('lift.pdf')
    plt.show()


def bar_plot():
    x = np.arange(3)
    bet_state_scores = [0.97, 0.97, 0.55, 0.17]
    dp_state_scores = [0.97, 0.92, 0.67, 0.23]
    bet_vision_scores = [0.82, 0.05, 0.07]
    dp_vision_scores = [0.7, 0.67, 0.7]
    bet_hybrid_scores = [0.92, 0.1, 0.27]
    dp_hybrid_scores = [0.72, 0.77, 0.6]

    bet_state_max_scores = [1, 1, 0.7, 0.4]
    dp_state_max_scores = [1, 1, 0.8, 0.45]
    bet_vision_max_scores = [1, 0.15, 0.2]
    dp_vision_max_scores = [1, 0.7, 0.8]
    bet_hybrid_max_scores = [0.95, 0.35, 0.4]
    dp_hybrid_max_scores = [1, 0.9, 0.85]

    bet_times = [0.67, 8.6, 10.32]
    dp_times = [5.22, 13.55, 12.37]

    bet_lift_horizons = {'S': 51, 'I': 44, 'H': 79}
    dp_lift_horizons = {'S': 66, 'I': 67, 'H': 51}
    bet_can_horizons = {'S': 120, 'I': 445, 'H': 376}
    dp_can_horizons = {'S': 127, 'I': 187, 'H': 182}
    bet_square_horizons = {'S': 228, 'I': 435, 'H': 357}
    dp_square_horizons = {'S': 225, 'I': 228, 'H': 218}
    bet_toolhang_horizons = {'S': 485}
    dp_toolhang_horizons = {'S': 475}
    pin = 0.25

    fig = plt.figure(figsize=(15, 4))

    # ax1 = fig.add_subplot(1, 2, 1)
    # x = np.arange(4)
    # ax1.bar(x - 0.11, bet_state_max_scores, 0.2, color='lightgray')
    # ax1.bar(x + 0.11, dp_state_max_scores, 0.2, color='lightsteelblue')
    # ax1.bar(x - 0.11, bet_state_scores, 0.2, label='miniBET', color='darkgray')
    # ax1.bar(x + 0.11, dp_state_scores, 0.2, label='DiffusionPolicy', color='steelblue')
    # ax1.set_title('State')
    # ax1.set_ylabel('Success Rate')
    # ax1.set_xticks(x, ['Lift', 'Can', 'Square', 'ToolHang'])
    # ax1.legend()
    #
    # ax2 = fig.add_subplot(1, 2, 2)
    # x = np.arange(3)
    # ax2.bar(x - 0.11, bet_vision_max_scores, 0.2, color='lightgray')
    # ax2.bar(x + 0.11, dp_vision_max_scores, 0.2, color='lightsteelblue')
    # ax2.bar(x - 0.11, bet_vision_scores, 0.2, label='miniBET', color='darkgray')
    # ax2.bar(x + 0.11, dp_vision_scores, 0.2, label='DiffusionPolicy', color='steelblue')
    # ax2.set_title('Image')
    # ax2.set_ylabel('Success Rate')
    # ax2.set_xticks(x, ['Lift', 'Can', 'Square'])
    # ax2.legend()

    # ax3 = fig.add_subplot(3, 2, 3)
    # x = np.arange(3)
    # ax3.bar(x - 0.11, bet_hybrid_max_scores, 0.2, color='lightgray')
    # ax3.bar(x + 0.11, dp_hybrid_max_scores, 0.2, color='lightsteelblue')
    # ax3.bar(x - 0.11, bet_hybrid_scores, 0.2, label='miniBET', color='darkgray')
    # ax3.bar(x + 0.11, dp_hybrid_scores, 0.2, label='DiffusionPolicy', color='steelblue')
    # ax3.set_title('Hybrid')
    # ax3.set_ylabel('Success Rate')
    # ax3.set_xticks(x, ['Lift', 'Can', 'Square'])
    # ax3.legend()
    #
    ax4 = fig.add_subplot(1, 3, 1)
    ax4.bar(x - 0.11, bet_times, 0.2, label='miniBET', color='darkgray')
    ax4.bar(x + 0.11, dp_times, 0.2, label='DiffusionPolicy', color='steelblue')
    ax4.set_title('Computation time')
    ax4.set_ylabel('Time')
    ax4.set_xticks(x, ['State', 'Image', 'Hybrid'])
    ax4.legend()

    def timeline(ax, idx, bet_horizons, dp_horizons):
        idx += 0.11
        ax.hlines(idx, 0, np.max(list(bet_horizons.values())), linewidth=10, color='darkgray', label='miniBET' if round(idx) == 3 else None)
        ax.vlines(list(bet_horizons.values()), idx - 0.1, idx + pin * np.ones(len(bet_horizons)), color='r')
        for (k, v) in bet_horizons.items():
            ax.annotate(
                k, xy=(v, idx + pin), ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor='r', boxstyle='circle')
            )
        idx -= 0.22
        ax.hlines(idx, 0, np.max(list(dp_horizons.values())), linewidth=10, color='steelblue', label='DiffusionPolicy' if round(idx) == 3 else None)
        ax.vlines(list(dp_horizons.values()), idx + 0.1, idx - pin * np.ones(len(dp_horizons)), color='r')
        for (k, v) in dp_horizons.items():
            ax.annotate(
                k, xy=(v, idx - pin), ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='r', boxstyle='circle')
            )

    ax5 = fig.add_subplot(1, 3, (2, 3))
    timeline(ax5, 3, bet_lift_horizons, dp_lift_horizons)
    timeline(ax5, 2, bet_can_horizons, dp_can_horizons)
    timeline(ax5, 1, bet_square_horizons, dp_square_horizons)
    timeline(ax5, 0, bet_toolhang_horizons, dp_toolhang_horizons)
    ax5.set_title('Frames required to complete the task')
    ax5.set_xlabel('Frames')
    ax5.set_yticks(np.arange(4), np.flip(['Lift', 'Can', 'Square', 'ToolHang']))
    ax5.legend()

    plt.tight_layout()
    plt.savefig('bar_plot.pdf')
    plt.show()


bar_plot()

# plot(bet_lift, bet_lift_scores, dp_lift, dp_lift_scores,
#      bet_lift_vision, bet_lift_vision_scores, dp_lift_vision, dp_lift_vision_scores)
     #bet_lift_hybrid, bet_lift_hybrid_scores, dp_lift_hybrid, dp_lift_hybrid_scores)
