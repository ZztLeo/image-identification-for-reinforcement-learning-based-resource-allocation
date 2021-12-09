import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from args import args


class DataGen(object):
    """
    Data Generation
    """

    def __init__(self):
        """
        different parameter follows different data distribution
        """
        self.min_bw_dist =
        self.max_bw_dist =
        self.recovery_time_dist =
        self.service_unavailable_dist =
        self.routing_stability_dist =
        self.max_delay_dist =
        self.jitter_dist =
        self.ber_dist =

    def genrate(self):
        """
        generate a random data following relative distribution
        :return:
        """

        return min_bw, max_bw, recovery_time, service_unavailable, routing_stability, max_delay, jitter_dist, ber_dist

    def generate_batch(self, batch_num: int=1):
        rtn = []
        for _ in range(batch_num):
            rtn.append(self.genrate())
        return rtn

class Generator(nn.Module):
    """
    Generator Module
    """
    def __init__(self, nn_nums: list=[1, 5, 1]):
        super(Generator, self).__init__()
        assert len(nn_nums) > 3
        last_index = len(nn_nums)-1
        self.out_num = nn_nums[last_index]
        self.model = nn.Sequential()
        for i in range(1, last_index):
            self.model.add_module(name="layer_{}".format(i),
                                  module=nn.Linear(in_features=nn_nums[i-1], out_features=nn_nums[i], bias=True))
            self.model.add_module(name="layer_{}_elu".format(i),
                                  module=nn.ELU(inplace=True))
        self.model.add_module(name="layer_{}".format(last_index),
                              module=nn.Linear(in_features=nn_nums[last_index-1], out_features=nn_nums[last_index], bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)


def main():
    """
    main method to run GAN in optical networks
    :return:
    """


if __name__ == "__main__":
    main()