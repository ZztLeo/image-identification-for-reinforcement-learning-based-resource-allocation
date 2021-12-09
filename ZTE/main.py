from matplotlib import use
use('Agg')
import os
from Service import RwaGame, ARRIVAL_NEWPORT, ARRIVAL_NOPORT
from model import MobileNetV2, SimpleNet, AlexNet, SqueezeNet, SimplestNet, ExpandSimpleNet, DeeperSimpleNet
from subproc_env import SubprocEnv
from storage import RolloutStorage
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
# from distributed_utils import dist_init, average_gradients, DistModule
from args import args
from utils import save_on_disk




def main():
    """
    主程序
    :return:
    """
    num_cls = args.wave_num * args.k + 1  # 所有的路由和波长选择组合，加上啥都不选
    action_shape = 1  # action的维度，默认是1.
    num_updates = int(args.steps) // args.workers // args.num_steps  # 梯度一共需要更新的次数
    if args.append_route.startswith("True"):
        channel_num = args.wave_num+args.k
    else:
        channel_num = args.wave_num


    # 解析weight
    if args.weight.startswith('None'):
        weight = None
    else:
        weight = args.weight
    # 创建actor_critic
    if args.mode.startswith('alg'):
        # ksp(args, weight)
        return
    elif args.mode.startswith('learning'):
        # CNN学习模式下，obs的shape应该是CHW
        obs_shape = (channel_num, args.img_height, args.img_width)
        if args.cnn.startswith('mobilenetv2'):
            actor_critic = MobileNetV2(in_channels=channel_num, num_classes=num_cls, t=6)
        elif args.cnn.startswith('simplenet'):
            actor_critic = SimpleNet(in_channels=channel_num, num_classes=num_cls)
        elif args.cnn.startswith('simplestnet'):
            actor_critic = SimplestNet(in_channels=channel_num, num_classes=num_cls)
        elif args.cnn.startswith('alexnet'):
            actor_critic = AlexNet(in_channels=channel_num, num_classes=num_cls)
        elif args.cnn.startswith('squeezenet'):
            actor_critic = SqueezeNet(in_channels=channel_num, num_classes=num_cls, version=1.0)
        elif args.cnn.startswith('expandsimplenet'):
            actor_critic = ExpandSimpleNet(in_channels=channel_num, num_classes=num_cls, expand_factor=args.expand_factor)
        elif args.cnn.startswith('deepersimplenet'):
            actor_critic = DeeperSimpleNet(in_channels=channel_num, num_classes=num_cls, expand_factor=args.expand_factor)
        else:
            raise NotImplementedError

        # 创建optimizer
        if args.algo.startswith("a2c"):
            optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.base_lr, eps=args.epsilon, alpha=args.alpha)
        elif args.algo.startswith("ppo"):
            optimizer = optim.Adam(actor_critic.parameters(), lr=args.base_lr, eps=args.epsilon)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if args.cuda.startswith("True"):
        # 如果要使用cuda进行计算
        actor_critic.cuda()
        # actor_critic = DistModule(actor_critic)

    # 判断是否是评估模式
    if args.evaluate:
        print("evaluate mode")
        models = {}
        times = 1
        prefix = "trained_models"
        directory = os.path.join(prefix, 'a2c', args.cnn, args.step_over)
        env = RwaGame(net_config=args.net, wave_num=args.wave_num, rou=args.rou, miu=args.miu,
                      max_iter=args.max_iter, k=args.k, mode=args.mode, img_width=args.img_width,
                      img_height=args.img_height, weight=weight, step_over=args.step_over)

        for model_file in reversed(sorted(os.listdir(directory), key=lambda item: int(item.split('.')[0]))):
            model_file = os.path.join(directory, model_file)
            print("evaluate model {}".format(model_file))
            params = torch.load(model_file)
            actor_critic.load_state_dict(params['state_dict'])
            actor_critic.eval()

            models[params['update_i']] = {}

            print("model loading is finished")
            for t in range(times):
                total_reward, total_services, allocated_services = 0, 0, 0
                obs, reward, done, info = env.reset()
                while not done:
                    inp = Variable(torch.Tensor(obs).unsqueeze(0), volatile=True)  # 禁止梯度更新
                    value, action, action_log_prob = actor_critic.act(inputs=inp, deterministic=True)  # 确定性决策
                    action = action.data.numpy()[0]
                    obs, reward, done, info = env.step(action=action[0])
                    total_reward += reward
                    if reward == ARRIVAL_NEWPORT or reward == ARRIVAL_NOPORT:
                        allocated_services += 1
                    if args.step_over.startswith('one_time'):
                        if info:
                            total_services += 1
                    elif args.step_over.startswith('one_service'):
                        total_services += 1
                    else:
                        raise NotImplementedError
                models[params['update_i']]['time'] = t
                models[params['update_i']]['reward'] = total_reward
                models[params['update_i']]['total_services'] = total_services
                models[params['update_i']]['allocated_services'] = allocated_services
                models[params['update_i']]['bp'] = (total_services-allocated_services)/total_services
        # 输出仿真结果
        # print("|updated model|test index|reward|bp|total services|allocated services|")
        # print("|:-----|:-----|:-----|:-----|:-----|:-----|")
        # for m in sorted(models):
            for i in range(times):
                print("|{up}|{id}|{r}|{bp:.4f}|{ts}|{als}|".format(up=params['update_i'],
                                                                  id=models[params['update_i']]['time'],
                                                                  r=models[params['update_i']]['reward'],
                                                                  bp=models[params['update_i']]['bp'],
                                                                  ts=models[params['update_i']]['total_services'],
                                                                  als=models[params['update_i']]['allocated_services']))
        return

    # 创建游戏环境
    envs = [make_env(net_config=args.net, wave_num=args.wave_num,
                     k=args.k, mode=args.mode, img_width=args.img_width,
                     img_height=args.img_height, weight=weight, step_over=args.step_over) for _ in range(args.workers)]
    envs = SubprocEnv(envs)
    # 创建游戏运行过程中相关变量存储更新的容器
    rollout = RolloutStorage(num_steps=args.num_steps, num_processes=args.workers,
                             obs_shape=obs_shape, action_shape=action_shape)
    current_obs = torch.zeros(args.workers, *obs_shape)

    observation, _, _, _ = envs.reset()
    update_current_obs(current_obs, observation, channel_num)

    rollout.observations[0].copy_(current_obs)
    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.workers, 1])
    final_rewards = torch.zeros([args.workers, 1])

    if args.cuda.startswith("True"):
        current_obs = current_obs.cuda()
        rollout.cuda()

    start = time.time()
    log_start = time.time()
    total_services = 0  # log_interval期间一共有多少个业务到达
    allocated_services = 0  # log_interval期间一共有多少个业务被分配成功
    update_begin = 0

    # 判断是否是接续之前的训练
    if args.resume:
        pms = torch.load(args.resume)
        actor_critic.load_state_dict(pms['state_dict'])
        optimizer.load_state_dict(pms['optimizer'])
        update_begin = pms['update_i']
        print("resume process from update_i {}, with base_lr {}".format(update_begin, args.base_lr))

    for updata_i in range(update_begin, num_updates):
        update_start = time.time()
        for step in range(args.num_steps):
            # 选择行为
            inp = Variable(rollout.observations[step], volatile=True)  # 禁止梯度更新
            value, action, action_log_prob = actor_critic.act(inputs=inp, deterministic=False)
           # print(action)
            # 压缩维度，放到cpu上执行。因为没有用到GPU，所以并没有什么卵用，权当提示
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            # 观察observation，以及下一个observation
            envs.step_async(cpu_actions)
            obs, reward, done, info = envs.step_wait()  # reward和done都是(n,)的numpy.ndarray向量
          #  if reward == ARRIVAL_NEWPORT_NEWPORT or reward == ARRIVAL_NOPORT_NEWPORT or reward == ARRIVAL_NOPORT_NOPORT:
           #     allocated_services += 1
            print(reward)
            for i in reward:
                if i == ARRIVAL_NEWPORT or i == ARRIVAL_NOPORT:
                    allocated_services += 1
          #  allocated_services += (reward==ARRIVAL_NEWPORT_NEWPORT or reward==ARRIVAL_NOPORT_NEWPORT or reward==ARRIVAL_NOPORT_NOPORT).any().sum()  # 计算分配成功的reward的次数
            # TODO 未解决
            if args.step_over.startswith('one_service'):
                total_services += (info==True).sum()  # 计算本次step中包含多少个业务到达事件
            # elif args.step_over.startswith('one_service'):
            #     total_services += args.workers
            else:
                raise NotImplementedError
            reward = torch.from_numpy(np.expand_dims(reward, 1)).float()
            episode_rewards += reward  # 累加reward分数
            # 如果游戏结束，则重新开始计算episode_rewards和final_rewards，并且以返回的reward为初始值重新进行累加。
            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])  # True --> 0, False --> 1
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
#            if done[len(done)-1]:
 #               print('游戏结束最终端口数量：',envs.get_all_edges_port())

            if args.cuda.startswith("True"):
                masks = masks.cuda()

            # 给masks扩充2个维度，与current_obs相乘。则运行结束的游戏进程对应的obs值会变成0，图像上表示全黑，即游戏结束的画面。
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
            update_current_obs(current_obs=current_obs, obs=obs, channel_num=channel_num)
            # 把本步骤得到的结果存储起来
            rollout.insert(step=step, current_obs=current_obs, action=action.data, action_log_prob=action_log_prob.data,
                           value_pred=value.data, reward=reward, mask=masks)

        # TODO 强行停止
        # envs.close()
        # return

        # 注意不要引用上述for循环定义的变量。下面变量的命名和使用都要注意。
        next_inp = Variable(rollout.observations[-1], volatile=True)  # 禁止梯度更新
        next_value = actor_critic(next_inp)[0].data  # 获取下一步的value值
        rollout.compute_returns(next_value=next_value, use_gae=False, gamma=args.gamma, tau=None)

        if args.algo.startswith('a2c'):
            # 下面进行A2C算法梯度更新
            inps = Variable(rollout.observations[:-1].view(-1, *obs_shape))
            acts = Variable(rollout.actions.view(-1, action_shape))

            # print("a2cs's acts size is {}".format(acts.size()))
            value, action_log_probs, cls_entropy = actor_critic.evaluate_actions(inputs=inps, actions=acts)
            print(cls_entropy.data)  
            # print("inputs' shape is {}".format(inps.size()))
            # print("value's shape is {}".format(value.size()))
            value = value.view(args.num_steps, args.workers, 1)
            # print("action_log_probs's shape is {}".format(action_log_probs.size()))
            action_log_probs = action_log_probs.view(args.num_steps, args.workers, 1)
            # 计算loss
            advantages = Variable(rollout.returns[:-1]) - value
            value_loss = advantages.pow(2).mean()  # L2Loss or MSE Loss
            action_loss = -(Variable(advantages.data) * action_log_probs).mean()
            total_loss = value_loss * args.value_loss_coef + action_loss - cls_entropy * args.entropy_coef

            optimizer.zero_grad()
            total_loss.backward()
            # 下面进行迷之操作。。梯度裁剪（https://www.cnblogs.com/lindaxin/p/7998196.html）
            nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
            # average_gradients(actor_critic)
            optimizer.step()
        elif args.algo.startswith('ppo'):
            # 下面进行PPO算法梯度更新
            advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            for e in range(args.ppo_epoch):
                data_generator = rollout.feed_forward_generator(advantages,
                                                                args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, actions_batch, \
                    return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, cls_entropy = actor_critic.evaluate_actions(
                        Variable(observations_batch),
                        Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()  # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()


        # 事后一支烟
        rollout.after_update()
        update_time = time.time() - update_start
        print("updates {} finished, cost time {}:{}".format(updata_i, update_time//60, update_time % 60))
        # print("total services is {}".format(total_services))
        # 存储模型
        if updata_i % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, 'a2c')
            save_path = os.path.join(save_path, args.cnn)
            save_path = os.path.join(save_path, args.step_over)
            if os.path.exists(save_path) and os.path.isdir(save_path):
                pass
            else:
                os.makedirs(save_path)
            save_file = os.path.join(save_path, str(updata_i)+'.tar')
            save_content = {
                'update_i': updata_i,
                'state_dict': actor_critic.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mean_reward': final_rewards.mean()
            }
            torch.save(save_content, save_file)

        # 输出日志
        if updata_i % args.log_interval == 0:
            end = time.time()
            interval = end - log_start
            remaining_seconds = (num_updates-updata_i-1) / args.log_interval * interval
            remaining_hours = int(remaining_seconds // 3600)
            remaining_minutes = int((remaining_seconds % 3600) / 60)
            total_num_steps = (updata_i+1) * args.workers * args.num_steps
            blocked_services = total_services - allocated_services
            bp = blocked_services / total_services
            wave_port_num, total_port_num = envs.get_all_edges_port()
            wave_occ_sum, resource_utilization_rate = envs.get_resourceUtilization()

            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, \
            entropy {:.5f}, value loss {:.5f}, policy loss {:.8f}, remaining time {}:{}, 阻塞率为{}/{}={}, \
                  各个波长端口数量为{}, 总的端口数量为{}, 带宽占用情况为{}, 资源占用率为{}".
                  format(updata_i, total_num_steps,
                         int(total_num_steps / (end - start)),
                         final_rewards.mean(),
                         final_rewards.median(),
                         final_rewards.min(),
                         final_rewards.max(), cls_entropy.data,
                         value_loss.data, action_loss.data,
                         remaining_hours, remaining_minutes,
                         blocked_services, total_services, bp,wave_port_num, total_port_num, wave_occ_sum,
                         resource_utilization_rate)
                         )
            # raise NotImplementedError
            total_services = 0
            allocated_services = 0
            log_start = time.time()

    envs.close()


def update_current_obs(current_obs, obs, channel_num):
    """
    全部更新当前的变量（不太明白源代码中为什么这么写？可能是跟fps有关吧，不能保证num_stack为抓取间隔）
    :param current_obs: 当前的observation
    :param obs: 要更新的observation
    """
    obs = torch.from_numpy(obs).float()
    current_obs[:, -channel_num:] = obs


def make_env(net_config: str, wave_num: int, 
             k: int, mode: str, img_width: int, img_height: int,
             weight, step_over):
    def _thunk():
        rwa_game = RwaGame(net_config=net_config, wave_num=wave_num, 
                           k=k, mode=mode, img_width=img_width,
                           img_height=img_height, weight=weight, step_over=step_over)
        return rwa_game
    return _thunk



if __name__ == "__main__":
    main()
