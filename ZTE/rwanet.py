import networkx as nx
from networkx import shortest_simple_paths
import os
import numpy as np
from draw import PltRwa
from args import args


class Vertex:
    def __init__(self, index):
        self.index = index
        self.port = None

    def is_port(self, path: tuple, target):
        self.port = [path, target, False]
        return self.port


class RwaNetwork(nx.Graph):

    def __init__(self, file_name, wave_num, append_route: bool=args.append_route.startswith("True"), k: int=args.k
                 , weight=None, file_prefix=args.file_prefix):
        super(RwaNetwork, self).__init__()
        self.net_name = file_name.split('.')[0]
        self.wave_num = wave_num
        self.append_route = append_route
        if append_route:
            print("open append-route option")
            self.k = k
            self.weight = weight
        else:
            self.k = 1
            self.weight = None
        filepath = os.path.join(file_prefix, file_name)
        # 读取拓扑md文件
        if os.path.isfile(filepath):
            datas = np.loadtxt(filepath, delimiter='|', skiprows=2, dtype=str)
            self.origin_data = datas[:, 1:(datas.shape[1] - 1)]
            # 内容行数
            for i in range(self.origin_data.shape[0]):
                wave_occ = [0 for _ in range(wave_num)]
                wave_avai = [True for i in wave_occ if i < 4]
                # None代表目标端口， False代表无端口
                port_avai = [{self.origin_data[i, 1]:False,self.origin_data[i, 2]:False} for _ in range(wave_num)]
                self.add_edge(self.origin_data[i, 1], self.origin_data[i, 2],
                              weight=float(self.origin_data[i, 3]), wave_occ=wave_occ,
                              is_wave_avai=wave_avai, port_avai=port_avai)
        else:
            raise FileExistsError("file {} doesn't exists.".format(filepath))
        self.draw = PltRwa(img_width=args.img_width, img_height=args.img_height, dpi=args.dpi,
                           line_width=args.line_width, node_size=args.node_size, net_name=file_name,
                           prefix=file_prefix)

    def gen_img(self, src: str, dst: str, mode: str) -> np.ndarray:
        """
        指定波长画端口还是路由画端口？
        :param src:
        :param dst:
        :param mode:
        :return:
        """
        if mode.startswith('alg'):
            return np.array([src, dst])
        elif mode.startswith('learning'):
            rtn = None
            for wave_index in range(self.wave_num):
                # 拓扑，源，宿，路由，目标端口节点，波长索引
                img = self.draw.draw(self, src, dst, None, None, wave_index)  # 画出指定波长拓扑
                if rtn is not None:
                    rtn = np.concatenate((rtn, img), axis=0)
                else:
                    rtn = np.array(img)
            # 判断是否将路由信息也放进去
            if self.append_route:  # 如果有路由信息，算路后，对每一条路画图
                if src is not None and dst is not None:
                    # k个点集
                    paths = self.k_shortest_paths(src, dst)
                    for nodes in paths:
                        target_units, _, _ = self.cost_port(nodes)
                    # 增加源宿点的用于业务上下路的端口
                        target_units.append(src)
                        target_units.append(dst)
                        img = self.draw.draw(self, src, dst, nodes, target_units, -1)
                        rtn = np.concatenate((rtn, img), axis=0)
            return rtn
        else:
            raise ValueError("wrong mode parameter")

    def k_shortest_paths(self, source, target):
        if source is None:
            return [None]
        generator = shortest_simple_paths(self, source, target, weight=self.weight)
        rtn = []
        index = 0
        for i in generator:
            index += 1
            if index > self.k:
                break
            rtn.append(i)
        return rtn

    def set_wave_state(self, wave_index, nodes: list, amount: int):
        """
        设置一条路径上的某个波长的可用状态
        :param wave_index: 编号从0开始
        :param nodes: 路径经过的节点序列
        :param amount: 带宽使用量
        :return:
        """
        assert len(nodes) >= 2
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            assert amount <= 4
           # print(self.get_edge_data(start_node, end_node)['wave_occ'][wave_index])
            if self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] is True:
                assert self.get_edge_data(start_node, end_node)['wave_occ'][wave_index] <= 4
                if amount < 4:
                    self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] = True
                else:
                    self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] = False
                self.get_edge_data(start_node, end_node)['wave_occ'][wave_index] = amount
                start_node = end_node
            else:
                raise Exception('NONONONONONONONONONON') 

    def get_avai_waves(self, nodes: list) -> list:
        """
        获取指定路径上的可用波长下标
        :param nodes: 路径经过的节点序列
        :return:
        """
        rtn = np.array([0 for _ in range(self.wave_num)])
        assert len(nodes) >= 2
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            rtn = np.logical_and(rtn,
                                 np.array(self.get_edge_data(start_node, end_node)['is_wave_avai']))
            start_node = end_node
        return np.where(rtn is True)[0].tolist()

    # def get_wave_occ(self, nodes: list, wave_index):
    #     """
    #     判断单条链路是否有业务存在
    #     :param nodes:
    #     :return:
    #     """
    #     paths = self.extract_path(nodes)
    #     for path in paths:
    #         if self.get_edge_data(path[0], path[1])['wave_occ'][wave_index] == 0:
    #     return self.get_edge_data(nodes[0], nodes[1])['wave_occ'][wave_index]

    def set_port_avai(self, path: tuple, target, is_avai, wave_index):
        """
        指定单条链路的指定位置设置端口信息
        :param path: 单条链路
        :param target: 目标网元
        :param is_avai: 端口状态
        :param wave_index: 波长索引
        :return:
        """
        if path[0] == target:
            self.get_edge_data(path[0], path[1])['port_avai'][wave_index][target] = is_avai
        elif path[1] == target:
            self.get_edge_data(path[0], path[1])['port_avai'][wave_index][target] = is_avai
        else:
            raise Exception('和目标端口相关的边信息对应错误')

    def get_port_avai(self, path: tuple, target):
        """
        返回当前链路的所有波长的端口信息
        :param path:
        :param target:
        :return:
        """
        assert path[0]==target or path[1]==target
        port_avai = []
        for i in range(self.wave_num):
            is_avai = self.get_edge_data(path[0], path[1])['port_avai'][i][target]
            port_avai.append(is_avai)
        return port_avai

    def exist_rw_allocation(self, path_list: list) -> [bool, int, int]:
        """
        扫描path_list中所有路径上的所有波长，按照FirstFit判断是否存在可分配方案
        :param path_list:
        :return: 是否存在路径，路径index，波长index
        """
        if len(path_list) == 0 or path_list[0] is None:
            return False, -1, -1

        for path_index, nodes in enumerate(path_list):
            # edges是[(),(),()]类型
            edges = self.extract_path(nodes)
            # print(edges)
            for wave_index in range(self.wave_num):
                is_avai = True
                for edge in edges:
                    if self.get_edge_data(edge[0], edge[1])['wave_occ'][wave_index] >= 4:
                        is_avai = False
                        break
                if is_avai is True:
                    return True, path_index, wave_index

        return False, -1, -1

    def is_allocable(self, path: list, wave_index: int) -> bool:
        """
        判断路由path上wave_index波长的路径是否可分配。
        :param path:
        :param wave_index:
        :return:
        """
        edges = self.extract_path(path)
        is_avai = True
        for edge in edges:
           # print('occupy of bandwidth:', self.get_edge_data(edge[0], edge[1])['wave_occ'][wave_index])
            if self.get_edge_data(edge[0], edge[1])['wave_occ'][wave_index] >= 4:
                is_avai = False
                break
        return is_avai

    def can_allocable(self, path: list, wave_index: int) -> bool:
        """
        判断路由path上wave_index波长的路径是否可分配。
        :param path:
        :param wave_index:
        :return:
        """
        edges = self.extract_path(path)
        is_avai = True
        for edge in edges:
           # print('occupy of bandwidth:', self.get_edge_data(edge[0], edge[1])['wave_occ'][wave_index])
            if self.get_edge_data(edge[0], edge[1])['wave_occ'][wave_index] > 4:
                is_avai = False
                break
        return is_avai


    def extract_path(self, nodes):
        assert len(nodes) >= 2
        rtn = []
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            rtn.append((start_node, end_node))
            start_node = end_node
        return rtn

    def cost_port(self, nodes):
        """
        对一条路径计算中继端口
        :param nodes: 数据类型为[,]
        为K条路径计算cost，若需要返回中继节点索引
        :return:
        """
        path = self.extract_path(nodes)
        target_units = []
        start_nodes = []
        end_nodes = []
        st_n = []
        en_n = []
        cost = 100
        for link in path:
            start_nodes.append(link[0])
            end_nodes.append(link[1])
            link_cost = self.get_edge_data(link[0], link[1])['weight']
            cost += link_cost
            cost += 50
            if cost > 800:
                target_units.append(link[0])
                cost = self.get_edge_data(link[0], link[1])['weight']
                cost += 50
                cost += 100
        for target_unit in target_units:
            if target_unit in start_nodes:
                st_n.append(target_unit)
            if target_unit in end_nodes:
                en_n.append(target_unit)
        return target_units, st_n, en_n

    def is_service_exist(self, link: tuple, wave_index):
        """
        判断指定索引上的一条链路上是否有其他业务存在，如果有业务存在查看链路是否有端口存在
        :param link: 指定路径
        :param wave_index:
        :return:返回当前业务情况以及链路两端端口情况
        """
        ser_exist = False
        start = False
        start_port = self.get_edge_data(link[0], link[1])['port_avai'][wave_index][link[0]]
        end = False
        end_port = self.get_edge_data(link[0], link[1])['port_avai'][wave_index][link[1]]
        if self.get_edge_data(link[0], link[1])['wave_occ'][wave_index] == 0:
            if start_port:
                start = True
                if end_port:
                    end = True
                    return ser_exist, start, end
                else:
                    return ser_exist, start, end
            else:
                if end_port:
                    end = True
                    return ser_exist, start, end
                else:
                    return ser_exist, start, end
        else:
            ser_exist = True
            if start_port:
                start = True
                if end_port:
                    end = True
                    return ser_exist, start, end
                else:
                    return ser_exist, start, end
            else:
                if end_port:
                    end = True
                    return ser_exist, start, end
                else:
                    return ser_exist, start, end

    def get_port_index(self, ports:list):
        """
        得到一条路径上每条链路起始点的端口索引
        :return:
        """
        index = []
        for i in range(len(ports)):
            if ports[i] is True:
                index.append(i)
       # print(index)
        return index

    def get_all_edges_port(self):
        """
        :return:得到所有波长的端口总数量和端口的总数量
        """
        all_edges = self.edges()
        port_sum = [0 for _ in self.wave_num]
        for edge in all_edges:
            port = self.get_port_avai(edge, edge[0])
            ports = self.get_port_avai(edge, edge[1])
            port_sum = [port_sum[i] + port[i] + ports[i] for i in range(len(self.wave_num))]
        return port_sum, sum(port_sum)

