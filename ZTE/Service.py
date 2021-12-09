
from rwanet import RwaNetwork
import numpy as np
from networkx import shortest_simple_paths
import random
from args import args
import xlrd
import utils
import os

modes = ['alg', 'learning']



# 重新开启一轮游戏
INIT = 0
# 没有可达的RW选项：
NOARRIVAL_NO = -1            # 选择No-Action选项
NOARRIVAL_OT = -1            # 选择其他RW选项
# 有可达的RW选项
ARRIVAL_NO = -2       # 选择No-Action选项
# 可达的不增加新端口的RW选项：
ARRIVAL_NOPORT = 2   # 选择不增加新端口的RW选项
# 可达的增加新端口的RW选项：
ARRIVAL_NEWPORT = 0      # 选择增加新端口的RW选项
# 可达的rw选项中选择了不可达的rw
ARRIVAL_OT = -1


class Service(object):
    def __init__(self, index: int, src: str, dst: str):
        super(Service, self).__init__()
        self.index = index
        self.src = src
        self.dst = dst
        self.bandwidth = 1

    def add_allocation(self, path: list, wave_index: int):
        self.path = path
        self.wave_index = wave_index


def cmp(x, y):
    if x[0] < y[0]:
        return -1
    if x[0] > y[0]:
        return 1
    return 0


class RwaGame(object):
    """
    RWA game, 模仿gym的实现
    """
    def __init__(self, net_config: str, wave_num: int,
                 k: int, mode: str, img_width: int, img_height: int,
                 weight, step_over: str='one_service'):
        """
        :param net_config: 网络配置文件
        :param wave_num: 链路波长数，CWDM是40， DWDM是80, 目前按照15波进行处理，可以设置对比试验
        :param mode: 模式，分为alg和learning两种，前者表示使用ksp+firstfit分配，后者表示使用rl算法学习
        :param img_width: 游戏界面的宽度
        :param img_height: 游戏界面的高度
        :param step_over: 步进的模式，one_time表示每调用一次step，执行一个时间步骤；one_service表示每调用一次step，执行到下一个service到达的时候。
        """
        super(RwaGame, self).__init__()
        print('创建RWA Game')
       # print(os.getpid())
        self.net_config = net_config
        self.wave_num = wave_num
        self.img_width = img_width
        self.img_height = img_height
        self.weight = weight
        # self.max_iter = max_iter
        self.k = k
        self.NO_ACTION = k*wave_num
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError("wrong mode parameter.")
        # 一旦游戏开始，iter和time都指向当前的event下标和时间点。
        self.service_index = 0
        # self.time = 0
        self.net = RwaNetwork(file_name=self.net_config, wave_num=self.wave_num)
        self.services = self.get_simulationService()
        # self.events = []  # time_point, service_index, is_arrival_event
        self.step_over = step_over

    # def gen_src_dst(self):
    #     nodes = list(self.net.nodes())
    #     assert len(nodes) > 1
    #     src_index = random.randint(0, len(nodes)-1)
    #     dst_index = random.randint(0, len(nodes)-1)
    #     while src_index == dst_index:
    #         dst_index = random.randint(0, len(nodes)-1)
    #     return nodes[src_index], nodes[dst_index]

    def again(self):
        """
        清空所有的状态缓存，将环境重置回开始之前的状态，但是业务序列不变
        :return:
        """
        self.service_index = 0
        ss = {}
        self.net = RwaNetwork(file_name=self.net_config, wave_num=self.wave_num)

        for val in self.services.values():
            serv = Service(val.index, val.src, val.dst)
            ss[val.index] = serv
        self.services = ss

        # 返回第一个业务请求的状态
        src, dst = self.services[0].src, self.services[0].dst
        observation = self.net.gen_img(src, dst, self.mode)
        reward = INIT
        done = False
        info = False
        # self.time = self.services[0].arrival_time
        return observation, reward, done, info

    def reset(self):
        """
        reset environment
        :return:
        """
        self.service_index = 0
        self.net = RwaNetwork(file_name=self.net_config, wave_num=self.wave_num)

        # for base_index in range(self.max_iter):
        #     src, dst = self.gen_src_dst()
        #     # arrival = np.random.poisson(lam=self.rou) + base_time + 1
        #     # leave = np.random.poisson(lam=self.miu) + arrival + 1
        #     self.services[base_index] = Service(base_index, src, dst)
        #     self.events.append([arrival, base_index, True])
        #     self.events.append([leave, base_index, False])
        #
        #     base_time = arrival
        # self.events.sort(key=lambda time: time[0])

        # 返回第一个业务请求的状态
        src, dst = self.services[self.service_index].src, self.services[self.service_index].dst
        observation = self.net.gen_img(src, dst, self.mode)
        reward = INIT
        done = False
        info = True
        return observation, reward, done, info

    def render(self):
        """
        渲染当前环境，返回当前环境的图像
        :return:
        """
        raise NotImplementedError

    def step(self, action) -> [object, float, bool, dict]:
        """
        根据self.step_over的设置，执行不同的step操作
        :param action:
        :return:
        """
        if self.step_over.startswith('one_service'):
            return self.step_one_service(action=action)

    def step_one_service(self, action)-> [object, float, bool, dict]:
        """
        在当前时间点self.time,执行行为action，获取reward，并且转向下一个业务。
        :param action: 所采取的行为，默认是int类型。如果取值为-1，表示暂停游戏，游戏状态不变化。
        :return:
        """

        if action == -1:
            return np.array([None, None]), 0, True, None

        done = False
        # info = False  # info表示本次是否处理业务
        # 如果当前业务为处理的最后一个业务，分配后直接游戏直接结束
        if self.service_index == len(self.services)-1:
            info = True
            ser = self.services[self.service_index]
            reward = self.exec_action(action, ser)
            observation = self.net.gen_img(None, None, self.mode)
            done = True
            port = self.get_all_edges_port()
            print('总端口数量为：', port)
            return observation, reward, done, info
        # 对于一般业务，执行选择的action后转向下一业务
        if self.service_index < len(self.services)-1:
            info = True
            current_ser = self.services[self.service_index]
            reward = self.exec_action(action, current_ser)
            self.service_index += 1
            next_ser = self.services[self.service_index]
            observation = self.net.gen_img(next_ser.src, next_ser.dst, self.mode)
            return observation, reward, done, info

    def exec_action(self, action: int, service: Service) -> float:
        """
        对到达的业务service，执行行为action，修改网络状态，并且返回reward。
        如果分配业务成功，则注意给service对象加入分配方案
        :param action:
        :param service:
        :return: reward
        """
      #  print('Action:', action)
        print('当前进程号：{},Action is {}'.format(os.getpid(), action))
        path_list = self.k_shortest_paths(service.src, service.dst)
   #     route_index = action // self.wave_num
    #    wave_index = action % self.wave_num
#        print(path_list)
        is_avai,_,_ = self.net.exist_rw_allocation(path_list)
        if action == self.NO_ACTION:
            # 待完善
            if is_avai:
                # 如果存在可分配的方案，但是选择了NO-ACTION
                return ARRIVAL_NO
            else:
                # 如果不存在可分配的方案，选择了NO-ACTION
                return NOARRIVAL_NO
        else:
            if is_avai:
                route_index = action // self.wave_num
                wave_index = action % self.wave_num
                if self.net.is_allocable(path_list[route_index], wave_index):      
                    path = self.net.extract_path(path_list[route_index])
               #    print('length of path is:', len(path))
                    port_sit = False
                    if self.net.get_port_avai(path[0], service.src)[wave_index] is False:
                        self.net.set_port_avai(path[0], service.src, True, wave_index)
                        port_sit = True
                    if self.net.get_port_avai(path[-1], service.dst)[wave_index] is False:
                        self.net.set_port_avai(path[-1], service.dst, True, wave_index)
                        port_sit = True
                    # 得到当前拓扑每条链路的业务状态以及端口信息
                    is_services = []
                    start_nodes = []
                    end_nodes = []
                    # service_index = []
                    ser_converge_change_situation = []
                    ser_relay_change_situation = []
                    # 得到所选路径的端口信息，curr包括True和False
                    _, curr_start_nodes, curr_end_nodes = self.net.cost_port(path_list[route_index])
                    # 得到一条路径中的每条链路端口的起始节点端口标号
                  #  curr_start_nodes = self.net.get_port_index(curr_start_nodes)
                  #  curr_end_nodes = self.net.get_port_index(curr_end_nodes)
                    for link in path:
                        # 得到拓扑中所选的整个路径的端口信息和业务情况
                        is_service, start_node, end_node = self.net.is_service_exist(link, wave_index)
                        is_services.append(is_service)
                        start_nodes.append(start_node)
                        end_nodes.append(end_node)
                        # 为选择的路径增加中继端口
                        if link[0] in curr_start_nodes or link[1] in curr_end_nodes:
                            if link[0] in curr_start_nodes:
                                if self.net.get_port_avai(link, link[0])[wave_index] is False:
                                    self.net.set_port_avai(link, link[0], True, wave_index)
                                    change = True
                                else:
                                    change = False 
                            else:
                                if self.net.get_port_avai(link, link[1])[wave_index] is False:
                                    self.net.set_port_avai(link, link[1], True, wave_index)
                                    change = True
                                else:
                                    change = False
                        else:
                            change = False
                        ser_relay_change_situation.append(change)
                    # 判断是否要进行业务汇聚，增加需要的端口
                    for i, is_exist in enumerate(is_services):
                    # if is_services[i] is True:
                    #     service_index.append(i)
                    # 需要进行业务汇聚，需要判断每条链路是否有业务.如果有业务判断此条链路是否有端口，没有端口需要再增加端口
                    # 若不存在业务则此条链路不需要增加端口
                  #  print('occupy of bandwidth:', self.net.get_edge_data(path[i][0], path[i][1])['wave_occ'][wave_index])
                        n = len(is_services)-1
                        if i == n and i == 0:
                            if is_exist is True: 
                                curr_bandwidth = self.net.get_edge_data(path[i][0], path[i][1])['wave_occ'][wave_index]
                                if self.net.get_port_avai(path[i], path[i][0])[wave_index] is True and  \
                                    self.net.get_port_avai(path[i], path[i][1])[wave_index] is True:
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth+1)
                                    change = False
                                else:
                                    if self.net.get_port_avai(path[i], path[i][0])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][0], True, wave_index)
                                    elif self.net.get_port_avai(path[i], path[i][1])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][1], True, wave_index)
                                    change = True
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth+1)
                                ser_converge_change_situation.append(change)
                            else:
                                assert is_exist is False
                                curr_bandwidth = self.net.get_edge_data(path[i][0], path[i][1])['wave_occ'][wave_index]
                                assert curr_bandwidth == 0
                                self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth+1)
                                change = False
                                ser_converge_change_situation.append(change)
                        else:
                            if i == 0 and is_exist is True:
                                assert len(path) >= 2
                                curr_bandwidth = self.net.get_edge_data(path[i][0], path[i][1])['wave_occ'][wave_index]
                                if self.net.get_port_avai(path[i], path[i][0])[wave_index] is True and  \
                                    self.net.get_port_avai(path[i], path[i][1])[wave_index] is True and \
                                        self.net.get_port_avai(path[i+1], path[i+1][0])[wave_index] is True:
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth+1)
                                    change = False
                                else:
                                    if self.net.get_port_avai(path[i], path[i][0])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][0], True, wave_index)
                                    elif self.net.get_port_avai(path[i], path[i][1])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][1], True, wave_index)
                                    elif self.net.get_port_avai(path[i+1], path[i+1][0])[wave_index] is False:
                                        self.net.set_port_avai(path[i+1], path[i+1][0], True, wave_index)
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth + 1)
                                    change = True
                                ser_converge_change_situation.append(change)
                            # 如果是最后一条链路
                            if i == n and is_exist is True and i != 0:
                                curr_bandwidth = self.net.get_edge_data(path[i][0], path[i][1])['wave_occ'][wave_index]
                                if self.net.get_port_avai(path[i], path[i][0])[wave_index] is True and \
                                    self.net.get_port_avai(path[i], path[i][1])[wave_index] is True and \
                                        self.net.get_port_avai(path[i-1], path[i-1][1])[wave_index] is True:
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth + 1)
                                    change = False
                                else:
                                    if self.net.get_port_avai(path[i], path[i][0])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][0], True, wave_index)
                                    elif self.net.get_port_avai(path[i], path[i][1])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][1], True, wave_index)
                                    elif self.net.get_port_avai(path[i-1], path[i-1][1])[wave_index] is False:
                                        self.net.set_port_avai(path[i-1], path[i-1][1], True, wave_index)
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth + 1)
                                    change = True
                                ser_converge_change_situation.append(change)
                            # 中间链路
                            if i != 0 and i != n and is_exist is True:
                                curr_bandwidth = self.net.get_edge_data(path[i][0], path[i][1])['wave_occ'][wave_index]
                                if self.net.get_port_avai(path[i], path[i][0])[wave_index] is True and \
                                    self.net.get_port_avai(path[i], path[i][1])[wave_index] is True and \
                                    self.net.get_port_avai(path[i - 1], path[i - 1][1])[wave_index] is True and \
                                        self.net.get_port_avai(path[i + 1], path[i + 1][0])[wave_index] is True:
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth + 1)
                                    change = False
                                else:
                                    if self.net.get_port_avai(path[i], path[i][0])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][0], True, wave_index)
                                    elif self.net.get_port_avai(path[i], path[i][1])[wave_index] is False:
                                        self.net.set_port_avai(path[i], path[i][1], True, wave_index)
                                    elif self.net.get_port_avai(path[i-1], path[i-1][1])[wave_index] is False:
                                        self.net.set_port_avai(path[i-1], path[i-1][1], True, wave_index)
                                    elif self.net.get_port_avai(path[i+1], path[i+1][0])[wave_index] is False:
                                        self.net.set_port_avai(path[i+1], path[i+1][0], True, wave_index)
                                    self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth + 1)
                                    change = True
                                ser_converge_change_situation.append(change)
                        # 如果一条链路没有业务存在
                            if is_exist is False:
                            # 设置波长状态
                                curr_bandwidth = self.net.get_edge_data(path[i][0], path[i][1])['wave_occ'][wave_index]
                                assert curr_bandwidth == 0
                                self.net.set_wave_state(wave_index, [path[i][0], path[i][1]], curr_bandwidth + 1)
                                change = False
                                ser_converge_change_situation.append(change)
                    # 将中继和业务汇聚端口情况进行整合
               # print('ser_relay_change_situation',ser_relay_change_situation) 
               # print('ser_converge_change_situation',ser_converge_change_situation)
                    assert len(ser_relay_change_situation) == len(ser_converge_change_situation)
                    total_change_situation = []
              #  print('realy change situation:',ser_relay_change_situation)
               # print('converge change situation:',ser_converge_change_situation)
                    for g in range(len(ser_relay_change_situation)):
                        if ser_relay_change_situation[g] == ser_converge_change_situation[g]:
                            total_change_situation.append(ser_converge_change_situation[g])
                        else:
                            total_change_situation.append(True)
                    total_change_situation.append(port_sit)
               # print('total change situation:',total_change_situation)
                    # 不需要增加新端口
                    if True not in total_change_situation:
                        assert self.net.can_allocable(path_list[route_index], wave_index) == True
                        service.add_allocation(path_list[route_index], wave_index)
                        return ARRIVAL_NOPORT
                    # 需要增加新端口
                    else:
                        assert self.net.can_allocable(path_list[route_index], wave_index) == True
                        service.add_allocation(path_list[route_index], wave_index)
                        return ARRIVAL_NEWPORT
                else:
                    return ARRIVAL_OT
            else:
                return NOARRIVAL_OT

    def k_shortest_paths(self, source, target):
        """
        如果源宿点是None，则返回len为1的None数组
        :param source:
        :param target:
        :return:
        """
        if source is None:
            return [None]
        generator = shortest_simple_paths(self.net, source, target, weight=self.weight)
        rtn = []
        index = 0
        for i in generator:
            index += 1
            if index > self.k:
                break
            rtn.append(i)
        return rtn

    def get_services(self):
        """
        得到所有的业务，每个业务按照[业务索引，源，宿]
        :return: {业务索引：业务[],...}
        """
        # 得到excel表中每行的[源，宿，业务数量]
        file = xlrd.open_workbook('/home/network/ZTE/Traffic Matrix.xlsx')
        sheet = file.sheet_by_name('Sheet1')
        i = 0
        datas = []
        file_services = {}
        for row in range(sheet.nrows):
            i += 1
            if i != 2 and i != 1:
                row_data = []
                j = 0
                for col in range(sheet.ncols):
                    j += 1
                    if j != 1:
                        cel = sheet.cell(row, col)
                        cel = str(cel)
                        content = cel.split(':')[1]
                        if len(content) > 2 and len(content) <= 4:
                            if content.startswith("'"):
                                content = content[1:-1]
                                # print(content)
                            else:
                                content = int(float(content))
                            row_data.append(content)
                datas.append(row_data)
        num = 0
        # 得到业务的字典{索引：Service(索引，源，宿)}
        for data in datas:
            for g in range(data[2]):
                file_services[num] = Service(num, data[0], data[1])
                num += 1
        return file_services

    def get_simulationService(self):
        index = 0
        services = {}
        # sort_index = []
        # tt = [(1, 6), (1, 8), (1, 9), (2, 7), (2, 9), (3, 6), (3, 7), (3, 8), (4, 9), (5, 3), (1, 5), (7, 5),
        #       (5, 9), (6, 7), (8, 3), [4, 6], [2, 8]]
        tt = [(1, 6), (1, 8), (2, 7), (2, 9), (3, 6), (3, 7), (3, 8), (4, 9), (5, 3), (1, 5), (7, 5),
              (5, 9), (6, 7), (8, 3), [4, 6]]
        for src, dst in tt:
            for _ in range(8):
#                print(index)
                src = str(src)
                dst = str(dst)
                services[index] = Service(index, src, dst)
                index += 1
        return services

    def get_all_edges_port(self):
        """
        :return:得到所有波长的端口总数量和端口的总数量
        """
        all_edges = self.net.edges()
        port_sum = [0 for _ in range(self.wave_num)]
        for edge in all_edges:
            port_start = self.net.get_port_avai(edge, edge[0])
            for i in range(len(port_start)):
                if port_start[i]:
                    port_start[i] = 1
                else:
                    port_start[i] = 0
            port_end = self.net.get_port_avai(edge, edge[1])
            for i in range(len(port_end)):
                if port_end[i]:
                    port_end[i] = 1
                else:
                    port_end[i] = 0
            port = [port_start[i] + port_end[i] for i in range(self.wave_num)]
            port_sum = [port_sum[i] + port[i] for i in range(self.wave_num)]
        return port_sum, sum(port_sum)

    def get_resourceUtilization(self):
        """
        得到整个网络的资源利用率
        :return:
        """
        wave_total_usage = [0 for _ in range(self.wave_num)]
        edge_num = len(self.net.edges)
        total_resource = self.wave_num * edge_num * len(self.net.draw.color)
        for edge in self.net.edges:
            usage = self.net.get_edge_data(edge[0], edge[1])['wave_occ']
            wave_total_usage = [wave_total_usage[i] + usage[i] for i in range(self.wave_num)]
        wave_occ_sum = sum(wave_total_usage)
        resource_utilization_rate = wave_occ_sum / total_resource
        return wave_occ_sum, resource_utilization_rate
