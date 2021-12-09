import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable as V
import torch.optim as optim
from BasicNet import SimpleNet, L2Loss
from wrapped_flappy_bird import GameState
from collections import deque
from torchvision.transforms import ToPILImage
from torchvision import transforms
import random
import sys

parser = argparse.ArgumentParser(
    description='PyTorch Flappy Bird by using DQN')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='decay rate of past observations')
parser.add_argument('--observe', type=int, default=1e4,  # TODO 为什么不直接用replay memory？
                    help='timesteps to observe before training. default 1e4')
parser.add_argument('--explore', type=int, default=3e6,
                    help='frames over which to anneal epsilon. After {explore} frames,'
                         ' eps is fixed with final-eps. default 3e6')
parser.add_argument('--final-eps', type=float, default=1e-4,
                    help='final value of epsilon')
parser.add_argument('--init-eps', type=float, default=1e-1,
                    help='epsilon initialization')
parser.add_argument('--trained-eps', type=float, default=1e-3,
                    help='fixed epsilon after explore+observe iterations in train mode')
parser.add_argument('--replay-memory', type=int, default=5e4,
                    help='number of previous transitions to remember')
parser.add_argument('--batch-size', type=int, default=32,
                    help='mini batch size')
parser.add_argument('--frame-per-action', type=int, default=1,
                    help='frames per action')
parser.add_argument('--action-num', type=int, default=2,
                    help='action space size.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-period', default=1e3, type=int,
                    help='iteration period to save trained model')
parser.add_argument('--save-path', metavar='PATH', type=str,
                    default='checkpoint', help='save path')


def main():
    """
    play flappy bird.
    """
    # write to file
    file = open('log.txt', 'w+')
    sys.stdout = file

    global args, best_prec1
    args = parser.parse_args()  # parse args

    # Step 1: build model network
    model = SimpleNet()

    # Step 2: initialize criterion and optimizer
    # criterion = nn.CrossEntropyLoss()  # TODO something in trouble
    criterion = L2Loss()
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-6)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Step 3: start training
    game_state = GameState()
    replay_memory = deque()
    eps = args.init_eps
    step = 0

    # input_actions[0] == 1: do nothing
    # input_actions[1] == 1: flap the bird
    do_nothing = np.zeros(args.action_num)
    do_nothing[0] = 1

    cur_img, _, _ = game_state.frame_step(do_nothing)  # at beginning, skip reward and is_terminated.
    cur_state = trans_img(cur_img)  # NxCxHxW = 4x80x80

    while(True):
        q_value = model(V(torch.Tensor(cur_state[np.newaxis, :])))  # TODO don't move it to GPU temporarily
        q_value = q_value.transpose(1, 0)  # shape of q_value is (1,2), convert it to (2,1)
        next_action = np.zeros([args.action_num])
        action_index = 0

        # choose an action
        if step % args.frame_per_action is 0:
            if random.random() <= eps:  # randomly select actions.
                action_index = random.randrange(args.action_num)
            else:  # select the action with max Q value
                _, action_index = torch.max(q_value, 0)
                action_index = action_index.data.numpy()[0]
            next_action[action_index] = 1
        else:
            next_action[0] = 1  # do nothing

        # scale down eps
        if step > args.observe + args.explore:
            eps = args.trained_eps
        elif eps > args.final_eps and step > args.observe:
            eps -= (args.init_eps - args.final_eps) / args.explore

        # do selected action, then observe next state, reward, and judge if game is terminated.
        next_img, next_reward, next_is_terminated = game_state.frame_step(next_action)
        next_state = trans_img(next_img)

        # store the transition from (cur_state, cur_reward, cur_is_terminated) --next_action-->
        # (next_state, next_reward, next_is_terminated)
        replay_memory.append((cur_state, next_action, next_reward, next_state, next_is_terminated))
        if len(replay_memory) > args.replay_memory:
            replay_memory.popleft()

        # if and only if observing enough transitions, do model update.
        if step > args.observe:
            batch_data = random.sample(replay_memory, args.batch_size)
            states = np.array([d[0] for d in batch_data])
            actions = np.array([d[1] for d in batch_data])
            rewards = np.array([d[2] for d in batch_data])
            next_states = np.array([d[3] for d in batch_data])
            ys = []

            cur_q_values = model(V(torch.Tensor(states)))  # Variable Nx2
            next_q_values = model(V(torch.Tensor(next_states)))  # Variable Nx2
            for i in range(0, args.batch_size):
                term = batch_data[i][4]
                # if game terminates, label equals reward; else, use bellman equation to update label
                if term:
                    ys.append(rewards[i])
                else:
                    ys.append(rewards[i] + args.gamma * np.max(next_q_values.data[i].numpy()))  # TODO bellman equation

            # calculate loss and execute updating
            q_actions = cur_q_values * V(torch.Tensor(actions))
            q_actions = torch.sum(q_actions, 1)

            loss = criterion(q_actions, V(torch.Tensor(ys)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(optimizer.state_dict())

        # udpate states and step
        step += 1
        cur_state = next_state

        # save progress every save_period iterations
        if step % args.save_period is 0:
            print("save checkpoint")
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, args.save_path + '_' + str(step) + '.tar')

        # print info
        state = ""
        if step <= args.observe:
            state = "observe"
        elif args.observe < step <= args.observe + args.explore:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP ", step,
              "| STATE ", state,
              "| EPSILON ", eps,
              "| ACTION ", action_index,
              "| REWARD ", next_reward,
              "| Q_MAX {:e}".format(q_value.data.max()),
              "| SCORE {:d}".format(game_state.score))


def trans_img(data: np.ndarray) -> np.ndarray:
    """
    convert img to Tensor with 4 channels.
    :param data: image with (512, 288, 3)
    :return:
    """
    to_pil = ToPILImage()
    resize = transforms.Resize((80, 80))

    img = to_pil(data)
    img = img.convert('L')  # convert to gray value
    img = resize(img)
    img = np.array(img)
    img = torch.Tensor(np.stack((img, img, img, img), axis=2))  # stack 4 img
    img = img.transpose(0, 2)
    return img.numpy()


if __name__ == "__main__":
    print("execute flappy bird!")
    main()
