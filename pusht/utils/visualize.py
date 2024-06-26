"""
Utils to postprocess video output.
"""
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PATH = '../results/'
n_steps = 300


def get_color(idx):
    viridis_cmap = plt.cm.get_cmap('viridis')
    n_colors = n_steps
    rgb = viridis_cmap(np.linspace(0, 1, n_colors))[:, :3] * 255
    return int(rgb[idx][2]), int(rgb[idx][1]), int(rgb[idx][0])


def track_end_effector(input_video, output_video, traj):
    cap = cv2.VideoCapture(input_video)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video, fourcc, 24, (512, 512))

    points = np.load(traj) * (512 / 96)

    for i in range(len(points) - 1):
        _, frame = cap.read()
        frame = cv2.resize(frame, (512, 512))

        for j in range(i):
            p1 = points[j]
            p2 = points[j + 1]
            cv2.line(
                frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), get_color(j), 3
            )
        writer.write(frame)
        cv2.imshow('', frame)
        if cv2.waitKey(20) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# def track_end_effector(filename, output):
#     cap = cv2.VideoCapture(filename)
#
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     writer = cv2.VideoWriter(output, fourcc, 24, (512, 512))
#
#     lower_red = (0, 50, 50)
#     upper_red = (50, 255, 255)
#     points = []
#
#     for _ in range(n_steps):
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame = cv2.resize(frame, (512, 512))
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, lower_red, upper_red)
#         cnts = cv2.findContours(
#             mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#         cnts = imutils.grab_contours(cnts)
#
#         if len(cnts) > 0:
#             area = max(cnts, key=cv2.contourArea)
#             (x, y), radius = cv2.minEnclosingCircle(area)
#             points.append((x, y))
#             if len(points) >= 2:
#                 for i in range(len(points) - 1):
#                     p1 = points[i]
#                     p2 = points[i + 1]
#                     cv2.line(
#                         frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), get_color(i), 3
#                     )
#         writer.write(frame)
#         cv2.imshow('', frame)
#         if cv2.waitKey(20) == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


def save_last_frame(filename, output):
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(output, frame)
    cap.release()


def save_first_frame(filename, output):
    cap = cv2.VideoCapture(filename)
    for _ in range(1):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(output, frame)
    cap.release()


def plot_loss():
    bet_state = np.load(PATH + 'mini_bet_state.npy')
    bet_vision = np.load(PATH + 'mini_bet_vision.npy')
    bet_hybrid = np.load(PATH + 'mini_bet_hybrid.npy')
    dp_state = np.load(PATH + 'dp_state.npy')
    dp_vision = np.load(PATH + 'dp_vision.npy')
    dp_hybrid = np.load(PATH + 'dp_hybrid.npy')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(bet_state[1:], label='state')
    ax1.plot(bet_vision[1:], label='image')
    ax1.plot(bet_hybrid[1:], label='hybrid')
    ax1.set_title('miniBET')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(dp_state[1:], label='state')
    ax2.plot(dp_vision[1:], label='image')
    ax2.plot(dp_hybrid[1:], label='hybrid')
    ax2.set_title('Diffusion Policy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('pusht_loss.pdf')
    plt.show()


def plot_score():
    bet_scores = [0.96, 0.07, 0.57]
    ibc_scores = [0.96, 0, 0.33]
    dp_scores = [0.99, 0.99, 0.99]
    bet_times = 60 * 0.001 * np.array([18, 7*60+52, 7*60+52])
    ibc_times = 60 * 0.001 * np.array([15, 0, 5*60+15])
    dp_times = 60 * 0.01 * np.array([18, 73, 74])
    x = np.arange(3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(x - 0.21, bet_scores, 0.2, label='miniBET', color='steelblue')
    ax1.bar(x, dp_scores, 0.2, label='Diffusion Policy', color='lightsteelblue')
    ax1.bar(x + 0.21, ibc_scores, 0.2, label='IBC', color='darkgray')
    ax1.set_title('Score')
    ax1.set_ylabel('mean reward')
    ax1.set_xticks(x, ['state', 'image', 'hybrid'])

    ax2.bar(x - 0.21, bet_times, 0.2, label='miniBET', color='steelblue')
    ax2.bar(x, dp_times, 0.2, label='Diffusion Policy', color='lightsteelblue')
    ax2.bar(x + 0.21, ibc_times, 0.2, label='IBC', color='darkgray')
    ax2.set_title('Computation time per epoch')
    ax2.set_ylabel('seconds')
    ax2.set_xticks(x, ['state', 'image', 'hybrid'])

    plt.legend()
    plt.tight_layout()
    plt.savefig('pusht_score3.pdf')
    plt.show()


def plot_history():
    bet_scores = 0.05 * np.array([0, 0, 8, 15, 14, 9, 1, 13, 17, 17, 18])
    dp_scores = 0.05 * np.array([0, 1, 3, 10, 16, 18, 17, 19, 19, 20, 20])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(np.arange(11), bet_scores, label='state', marker='x', color='r',)
    ax1.plot(bet_scores, linestyle='--')
    ax1.grid(axis='y')
    ax1.set_title('miniBET')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Success Ratio')

    def x100(x, pos):
        return f'{x * 100:.0f}'
    ax1.xaxis.set_major_formatter(FuncFormatter(x100))
    ax1.legend()

    ax2.scatter(np.arange(11), dp_scores, marker='x', color='r', )
    ax2.plot(dp_scores, linestyle='--')
    ax2.grid(axis='y')
    ax2.set_title('Diffusion Policy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Success Ratio')

    def x10(x, pos):
        return f'{x * 10:.0f}'
    ax2.xaxis.set_major_formatter(FuncFormatter(x10))
    ax2.legend()

    plt.tight_layout()
    plt.savefig('pusht_history.pdf')
    plt.show()


def plot():
    bet_state = np.load('../results/mini_bet_state.npy')
    bet_vision = np.load(PATH + 'mini_bet_vision.npy')
    bet_hybrid = np.load(PATH + 'mini_bet_hybrid.npy')
    ibc_state = np.load('../results/ibc_state_3.npy')
    dp_state = np.load(PATH + 'dp_state.npy')
    dp_vision = np.load(PATH + 'dp_vision.npy')
    dp_hybrid = np.load(PATH + 'dp_hybrid.npy')

    def x10(x, pos):
        return f'{x * 10:.0f}'

    def x100(x, pos):
        return f'{x * 100:.0f}'

    bet_scores = 0.05 * np.array([0, 0, 8, 15, 14, 9, 1, 13, 17, 17, 18])
    bet_v_scores = np.zeros(11)
    # ibc_scores = 0.05 * np.array([0, 1, 7, 9, 19, 17, 15, 17, 17, 18, 18])
    ibc_scores = 0.05 * np.array([0, 0, 0, 4, 4, 12, 12, 18, 19, 20, 20])
    dp_scores = 0.05 * np.array([0, 1, 3, 10, 16, 18, 17, 19, 19, 20, 20])
    dp_v_scores = 0.05 * np.array([0, 0, 9, 17, 15, 16, 19, 17, 14, 16, 16])

    fig, ax = plt.subplots(3, 2, figsize=(10, 12))

    ax[0][0].plot(bet_state[10:], label='state')
    ax[0][0].plot(bet_vision[10:], label='image')
    # ax[0][0].plot(bet_hybrid[1:], label='hybrid')
    ax[0][0].set_title('miniBET')
    ax[0][0].set_xlabel('Epoch')
    ax[0][0].set_ylabel('Loss')
    ax[0][0].legend()

    ax[0][1].plot(bet_scores, linestyle='--')
    ax[0][1].plot(bet_v_scores, linestyle='--')
    ax[0][1].scatter(np.arange(11), bet_scores, label='state', marker='x', color='#1a5c8d')
    ax[0][1].scatter(np.arange(11), bet_v_scores, label='image', marker='x', color='#cc6600')
    ax[0][1].grid(axis='y')
    ax[0][1].set_title('miniBET')
    ax[0][1].set_xlabel('Epoch')
    ax[0][1].set_ylabel('Success Rate')
    ax[0][1].xaxis.set_major_formatter(FuncFormatter(x100))
    ax[0][1].legend()

    ax[1][0].plot(ibc_state[10:], label='state')
    ax[1][0].set_title('IBC')
    ax[1][0].set_xlabel('Epoch')
    ax[1][0].set_ylabel('Loss')
    ax[1][0].legend()

    ax[1][1].plot(ibc_scores, linestyle='--')
    ax[1][1].scatter(np.arange(11), ibc_scores, label='state', marker='x', color='#1a5c8d')
    ax[1][1].grid(axis='y')
    ax[1][1].set_title('IBC')
    ax[1][1].set_xlabel('Epoch')
    ax[1][1].set_ylabel('Success Rate')
    ax[1][1].xaxis.set_major_formatter(FuncFormatter(x100))
    ax[1][1].legend()

    ax[2][0].plot(dp_state[1:], label='state')
    ax[2][0].plot(dp_vision[1:], label='image')
    # ax[2][0].plot(dp_hybrid[1:], label='hybrid')
    ax[2][0].set_title('Diffusion Policy')
    ax[2][0].set_xlabel('Epoch')
    ax[2][0].set_ylabel('Loss')
    ax[2][0].legend()

    ax[2][1].plot(dp_scores, linestyle='--')
    ax[2][1].plot(dp_v_scores, linestyle='--')
    ax[2][1].scatter(np.arange(11), dp_scores, label='state', marker='x', color='#1a5c8d')
    ax[2][1].scatter(np.arange(11), dp_v_scores, label='image', marker='x', color='#cc6600')
    ax[2][1].grid(axis='y')
    ax[2][1].set_title('Diffusion Policy')
    ax[2][1].set_xlabel('Epoch')
    ax[2][1].set_ylabel('Success Rate')
    ax[2][1].xaxis.set_major_formatter(FuncFormatter(x10))
    ax[2][1].legend()

    plt.tight_layout()
    plt.savefig('pusht.pdf')
    plt.show()


# plt.plot(np.load(PATH + 'dp_kan.npy')[1:])
# plt.plot(np.load(PATH + 'dp_state.npy')[1:])
# plt.show()

track_end_effector(PATH + 'dp_kan.mp4', PATH + 'dp_kan_tr.mp4', PATH + 'dp_kan_history.npy')

# save_last_frame(PATH + 'mini_bet_state_2_tr', PATH + 'mini_bet_state_2.png')
# plot_score()

# plot_history()
# plt.plot(np.load('../results/ibc_state.npy')[1:])
# plt.plot(np.load('../results/ibc_hybrid_old.npy')[1:])
# plt.show()
