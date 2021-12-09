import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as pt
import numpy as np
import networkx as nx
import os
from io import BytesIO
from PIL import Image
import time


class RwaNetwork(nx.Graph):
    """
    RWA network
    """

    def __init__(self, filename: str, wave_num: int, append_route: bool = True, weight=None,
                 file_prefix='E:/graduate student/实验室/中兴-业务编排/test'):
        """

        :param filename: 标明网络的md文件，其中前两行是表头和md表格的标志“|:---|”，内容为index，src，dst，weight
        :param wave_num: 每条链路包含的波长个数
        :param append_route: 在生成网络图像的时候，是否把路由信息也加进去
        :param k: 如果加路由信息，则加几个路由
        """
        super(RwaNetwork, self).__init__()
        self.net_name = filename.split('.')[0]
        self.wave_num = wave_num
        self.append_route = append_route
        self.k = 1
        self.weight = None
        filepath = os.path.join(file_prefix, filename)
        if os.path.isfile(filepath):
            datas = np.loadtxt(filepath, delimiter='|', skiprows=2, dtype=str)
            self.origin_data = datas[:, 1:(datas.shape[1] - 1)]
            for i in range(self.origin_data.shape[0]):
                wave_avai = [True for i in range(wave_num)]
                self.add_edge(self.origin_data[i, 1], self.origin_data[i, 2],
                              weight=float(self.origin_data[i, 3]),
                              is_wave_avai=wave_avai)
        else:
            raise FileExistsError("file {} doesn't exists.".format(filepath))





class PltRwa(object):

    def __init__(self, img_width=224, img_height=224, dpi=72, line_width=1, node_size=0.1,
                 net_name: str="7node_10link.md"):
        """
        这里的生成图像高度和宽度是指输出的图像维度，而不是从matplotlib生成的图像的shape。从matplotlib生成的图像的shape目前来说无法精确控制，
        只能大致划定范围，原因是在生成图像以后，需要去除padding，因此生成的图像会略小于要求的宽高
        :param img_width: 生成图像的宽度
        :param img_height: 生成图像的高度
        :param dpi: 生成图像的dpi
        """
        self.img_width = img_width
        self.img_height = img_height
        self.dpi = dpi
        self.line_width = line_width
        self.node_size = node_size
        # self.file_prefix = prefix
        self.layout = self.get_layout("9node_14link_layout6.md")  # 获取布局参数
        self.node_num = len(self.layout)

    def draw1(self, net: nx.Graph, src: str, dst: str, bandwidth: str, target_links:list, target_units:list,
              path: None, wave_index: int = -1):

        # 首先画出图像
        fig, ax = plt.subplots()
        fig.set_size_inches(self.img_width / self.dpi, self.img_height / self.dpi)
        fig.set_dpi(self.dpi)
        fig.set_frameon(False)
        ax.axis('off')
        ax.set_xlim(left=0, right=1)  # 当前画图区域边界
        ax.set_ylim(bottom=0, top=1)
        bias = self.node_size / 2

        def draw_port(node1, node2):
            locs = [self.layout[node1], self.layout[node2]]
            c = []
            for items in locs:
                b = []
                for item in items:
                    b.append(item + self.node_size / 2)
                c.append(b)
            try:
                if c[1][0] >= c[0][0]:
                    tan = (c[1][1] - c[0][1]) / (c[1][0] - c[0][0])
                    if tan > 1 or tan < -1:
                        b = c[1][1] - tan * c[1][0]
                        if c[1][1] >= c[0][1]:
                            x0 = (c[1][1] - self.node_size / 2 - b) / tan
                            port = pt.Circle(xy=(x0, c[1][1] - self.node_size / 2),
                                             radius=self.node_size / 9, facecolor='r')
                            ax.add_patch(port)
                        else:
                            x0 = (c[1][1] + self.node_size / 2 - b) / tan
                            port = pt.Circle(xy=(x0, c[1][1] + self.node_size / 2),
                                             radius=self.node_size / 9, facecolor='r')
                            ax.add_patch(port)
                    else:
                        b = c[1][1] - tan * c[1][0]
                        # if c[1][1] >= c[0][1]:
                        y0 = (c[1][0] - self.node_size / 2) * tan + b
                        port = pt.Circle(xy=(c[1][0] - self.node_size / 2, y0),
                                         radius=self.node_size / 9, facecolor='r')
                        ax.add_patch(port)
                else:
                    tan = (c[1][1] - c[0][1]) / (c[1][0] - c[0][0])
                    if tan > 1 or tan < -1:
                        b = c[1][1] - tan * c[1][0]
                        if c[1][1] >= c[0][1]:
                            x0 = (c[1][1] - self.node_size / 2 - b) / tan
                            port = pt.Circle(xy=(x0, c[1][1] - self.node_size / 2),
                                             radius=self.node_size / 9, facecolor='r')
                            ax.add_patch(port)
                        else:
                            x0 = (c[1][1] + self.node_size / 2 - b) / tan
                            port = pt.Circle(xy=(x0, c[1][1] + self.node_size / 2),
                                             radius=self.node_size / 9, facecolor='r')
                            ax.add_patch(port)
                    else:
                        b = c[1][1] - tan * c[1][0]
                        # if c[1][1] >= c[0][1]:
                        y0 = (c[1][0] + self.node_size / 2) * tan + b
                        port = pt.Circle(xy=(c[1][0] + self.node_size / 2, y0),
                                         radius=self.node_size / 9, facecolor='r')
                        ax.add_patch(port)
            except:
                tan = self.node_size / 2
                if c[1][1] >= c[0][1]:
                    port = pt.Circle(xy=(c[1][0], c[1][1] - tan),
                                     radius=self.node_size / 9, facecolor='r')
                    ax.add_patch(port)
                else:
                    port = pt.Circle(xy=(c[1][0], c[1][1] + tan),
                                     radius=self.node_size / 9, facecolor='r')
                    ax.add_patch(port)
        if path is None:
            # 如果没有路径，则画出全网指定波长的图像
            assert wave_index >= 0
            # 先把所有节点画出来
            for name, loc in self.layout.items():
                if src is not None and ((name == src) or (name == dst)):
                    shp = pt.Circle(xy=(loc[0] + bias, loc[1] + bias), radius=self.node_size / 2, facecolor='c'
                                    , fill=False)
                else:
                    shp = pt.Rectangle((loc[0], loc[1]), self.node_size, self.node_size, linewidth=0.5, fill=False,
                                       facecolor='black')
                ax.add_patch(shp)

            for edge in net.edges:
                locs = [self.layout[edge[0]], self.layout[edge[1]]]
                c = []
                for items in locs:
                    b = []
                    for item in items:
                        b.append(item + self.node_size / 2)
                    c.append(b)
                for target_link in target_links:
                    for target_unit in target_units:
                        if (edge[0] == target_link[0] and edge[1] == target_link[1]) or (edge[0] == target_link[1] and edge[1] == target_link[0]):
                            if target_link[0] == target_unit:
                                draw_port(target_link[1], target_unit)
                            elif target_link[1] == target_unit:
                                draw_port(target_link[0], target_unit)
                    # 得到x,y坐标
                    line_xs, line_ys = zip(*locs)
                    line_xs = [i + bias for i in line_xs]
                    line_ys = [i + bias for i in line_ys]
                    ax.add_line(Line2D(line_xs, line_ys, linewidth=0.5, color='black'))
                plt.subplots_adjust(left=0.00, right=1.00, wspace=0.25, hspace=0.25, bottom=0.00, top=1.00)
            # plt.show()

        else:
            # 如果有路由，则画出路由经过的节点和链路图像
            assert len(path) >= 2
            assert wave_index == -1
            # 把经过的所有节点画出来
            for node in path:
                loc = self.layout[node]
                if node.startswith(src) or node.startswith(dst):
                    shp = pt.Circle(xy=(loc[0] + bias, loc[1] + bias), radius=self.node_size / 2, facecolor='gold')
                else:
                    shp = pt.Rectangle((loc[0], loc[1]), self.node_size, self.node_size, linewidth=0, edgecolor='r',
                                       facecolor='black')
                ax.add_patch(shp)
            # 把经过的所有链路画出来
            start_node = path[0]
            for i in range(1, len(path)):
                locs = [self.layout[start_node], self.layout[path[i]]]
                line_xs, line_ys = zip(*locs)
                line_xs = [i + bias for i in line_xs]
                line_ys = [i + bias for i in line_ys]
                ax.add_line(Line2D(line_xs, line_ys, linewidth=1, color='blue'))
                start_node = path[i]


        # 申请缓冲地址
        buffer_ = BytesIO()  # using buffer,great way!
        # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
        plt.savefig(buffer_, format="png", transparent="True", pad_inches=0, bbox_inches='tight')
        buffer_.seek(0)
        # 用PIL或CV2从内存中读取
        dataPIL = Image.open(buffer_)
        dataPIL = dataPIL.convert('L')
        dataPIL = dataPIL.resize(size=(self.img_width, self.img_height))
        # TODO 临时添加
        dataPIL.save(str(wave_index) + ".png", directory='MNIST_data')
        plt.show()




    def draw(self, net: nx.Graph, src: str, dst: str, path: None, wave_index: int = -1):
        """
        坐标从左下角[0,0]到右上角[img_width, img_height]
        :param net: RwaNetwork类型
        :param src: 源点
        :param dst: 宿点
        :param path: 源宿点经过的路由节点列表，如果非None，表示仅仅画出路由路线，此时忽略wave_index
        :param wave_index: -1表示直接画全网拓扑，其他值表示画网络指定波长的拓扑。
        :return:
        """
        # 首先画出图像
        fig, ax = plt.subplots()
        fig.set_size_inches(self.img_width / self.dpi, self.img_height / self.dpi)
        fig.set_dpi(self.dpi)
        fig.set_frameon(False)
        ax.axis('off')
        ax.set_xlim(left=0, right=1)  # 当前画图区域边界
        ax.set_ylim(bottom=0, top=1)
        bias = self.node_size / 2

        if path is None:
            # 如果没有路径，则画出全网指定波长的图像
            assert wave_index >= 0
            # 先把所有节点画出来
            for name, loc in self.layout.items():
                if src is not None and ((name == src) or (name == dst)):
                    shp = pt.Circle(xy=(loc[0] + bias, loc[1] + bias), radius=self.node_size / 2, facecolor='c'
                                    ,fill=False)
                else:
                    shp = pt.Rectangle((loc[0], loc[1]), self.node_size, self.node_size, linewidth=0.5,fill=False,
                                       facecolor='black')
                    # port = pt.Circle(xy=(loc[0] + self.node_size/2, loc[1]), radius=self.node_size / 9, facecolor='r')
                    # ax.add_patch(port)
                ax.add_patch(shp)
            print(len(net.edges))
            for edge in net.edges:
                if net.get_edge_data(edge[0], edge[1])["is_wave_avai"][wave_index] is True:
                    # 如果在该波长层面上该链路可用
                    locs = [self.layout[edge[0]], self.layout[edge[1]]]
                    c = []
                    for items in locs:
                        b = []
                        for item in items:
                            b.append(item+self.node_size/2)
                        c.append(b)
                    print(c)
                    try :
                        if c[1][0] >= c[0][0]:
                            tan = (c[1][1] - c[0][1]) / (c[1][0] - c[0][0])
                            if tan > 1 or tan < -1:
                                b = c[1][1] - tan * c[1][0]
                                if c[1][1] >= c[0][1]:
                                    x0 = (c[1][1] - self.node_size/2 - b)/tan
                                    x1 = (c[0][1] + self.node_size/2 - b)/tan
                                    port = pt.Circle(xy=(x0, c[1][1] - self.node_size/2 ),
                                                     radius=self.node_size / 9, facecolor='r')
                                    port1 = pt.Circle(xy=(x1,c[0][1] + self.node_size/2),
                                              radius=self.node_size / 9, facecolor='r')
                                    ax.add_patch(port)
                                    ax.add_patch(port1)
                                else:
                                    x0 = (c[1][1] + self.node_size / 2 - b) / tan
                                    x1 = (c[0][1] - self.node_size / 2 - b) / tan
                                    port = pt.Circle(xy=(x0, c[1][1] + self.node_size / 2),
                                                     radius=self.node_size / 9, facecolor='r')
                                    port1 = pt.Circle(xy=(x1, c[0][1] - self.node_size / 2),
                                                      radius=self.node_size / 9, facecolor='r')
                                    ax.add_patch(port)
                                    ax.add_patch(port1)
                            else:
                                b = c[1][1] - tan * c[1][0]
                                # if c[1][1] >= c[0][1]:
                                y0 = (c[1][0] - self.node_size / 2) * tan + b
                                y1 = (c[0][0] + self.node_size / 2) * tan + b
                                port = pt.Circle(xy=(c[1][0] - self.node_size / 2, y0),
                                                 radius=self.node_size / 9, facecolor='r')
                                port1 = pt.Circle(xy=(c[0][0] + self.node_size / 2, y1),
                                                  radius=self.node_size / 9, facecolor='r')
                                ax.add_patch(port)
                                ax.add_patch(port1)
                                # else:
                                #     y0 = (c[1][0] - self.node_size / 2) * tan + b
                                #     y1 = (c[0][0] + self.node_size / 2) * tan + b
                                #     port = pt.Circle(xy=(c[1][0] - self.node_size / 2, y0),
                                #                      radius=self.node_size / 9, facecolor='r')
                                #     port1 = pt.Circle(xy=(c[0][0] + self.node_size / 2, y1),
                                #                       radius=self.node_size / 9, facecolor='r')
                        else:
                            tan = (c[1][1] - c[0][1]) / (c[1][0] - c[0][0])
                            if tan > 1 or tan < -1:
                                b = c[1][1] - tan * c[1][0]
                                if c[1][1] >= c[0][1]:
                                    x0 = (c[1][1] + self.node_size / 2 - b) / tan
                                    x1 = (c[0][1] - self.node_size / 2 - b) / tan
                                    port = pt.Circle(xy=(x0, c[1][1] + self.node_size / 2),
                                                     radius=self.node_size / 9, facecolor='r')
                                    port1 = pt.Circle(xy=(x1, c[0][1] - self.node_size / 2),
                                              radius=self.node_size / 9, facecolor='r')
                                    ax.add_patch(port)
                                    ax.add_patch(port1)
                                else:
                                    x0 = (c[1][1] + self.node_size / 2 - b) / tan
                                    x1 = (c[0][1] - self.node_size / 2 - b) / tan
                                    port = pt.Circle(xy=(x0, c[1][1] + self.node_size / 2),
                                                     radius=self.node_size / 9, facecolor='r')
                                    port1 = pt.Circle(xy=(x1, c[0][1] - self.node_size / 2),
                                                      radius=self.node_size / 9, facecolor='r')
                                    ax.add_patch(port)
                                    ax.add_patch(port1)
                            else:
                                b = c[1][1] - tan * c[1][0]
                                # if c[1][1] >= c[0][1]:
                                y0 = (c[1][0] + self.node_size / 2) * tan + b
                                y1 = (c[0][0] - self.node_size / 2) * tan + b
                                port = pt.Circle(xy=(c[1][0] + self.node_size / 2 , y0),
                                                 radius=self.node_size / 9, facecolor='r')
                                port1 = pt.Circle(xy=(c[0][0] - self.node_size / 2, y1),
                                                radius=self.node_size / 9, facecolor='r')
                                ax.add_patch(port)
                                ax.add_patch(port1)
                                # else:
                                #     y0 = (c[1][0] + self.node_size / 2) * tan + b
                                #     y1 = (c[0][0] - self.node_size / 2) * tan + b
                                #     port = pt.Circle(xy=(c[1][0] + self.node_size / 2, y0),
                                #                      radius=self.node_size / 9, facecolor='r')
                                #     port1 = pt.Circle(xy=(c[0][0] - self.node_size / 2, y1),
                                #                       radius=self.node_size / 9, facecolor='r')
                        print(tan)

                    except:
                        tan = self.node_size/2
                        if c[1][1] >= c[0][1]:
                            port = pt.Circle(xy=(c[1][0], c[1][1] - tan),
                                     radius=self.node_size / 9, facecolor='r')
                            port1 = pt.Circle(xy=(c[0][0], c[0][1] + tan),
                                         radius=self.node_size / 9, facecolor='r')
                            ax.add_patch(port)
                            ax.add_patch(port1)
                        else:
                            port = pt.Circle(xy=(c[1][0], c[1][1] + tan),
                                         radius=self.node_size / 9, facecolor='r')
                            port1 = pt.Circle(xy=(c[0][0], c[0][1] - tan),
                                             radius=self.node_size / 9, facecolor='r')
                            ax.add_patch(port)
                            ax.add_patch(port1)
                    # 得到x,y坐标
                    line_xs, line_ys = zip(*locs)
                    line_xs = [i + bias for i in line_xs]
                    line_ys = [i + bias for i in line_ys]
                    ax.add_line(Line2D(line_xs, line_ys, linewidth=0.5, color='black'))
            plt.subplots_adjust(left=0.00, right=1.00, wspace=0.25, hspace=0.25, bottom=0.00, top=1.00)
            # plt.show()
        else:
            # 如果有路由，则画出路由经过的节点和链路图像
            assert len(path) >= 2
            assert wave_index == -1
            # 把经过的所有节点画出来
            for node in path:
                loc = self.layout[node]
                if node.startswith(src) or node.startswith(dst):
                    shp = pt.Circle(xy=(loc[0] + bias, loc[1] + bias), radius=self.node_size / 2, facecolor='gold')
                else:
                    shp = pt.Rectangle((loc[0], loc[1]), self.node_size, self.node_size, linewidth=0, edgecolor='r',
                                       facecolor='black')
                ax.add_patch(shp)
            # 把经过的所有链路画出来
            start_node = path[0]
            for i in range(1, len(path)):
                locs = [self.layout[start_node], self.layout[path[i]]]
                line_xs, line_ys = zip(*locs)
                line_xs = [i + bias for i in line_xs]
                line_ys = [i + bias for i in line_ys]
                ax.add_line(Line2D(line_xs, line_ys, linewidth=1, color='blue'))
                start_node = path[i]

        # 申请缓冲地址
        buffer_ = BytesIO()  # using buffer,great way!
        # 保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
        plt.savefig(buffer_, format="png", transparent="True", pad_inches=0, bbox_inches='tight')
        buffer_.seek(0)
        # 用PIL或CV2从内存中读取
        dataPIL = Image.open(buffer_)
        dataPIL = dataPIL.convert('L')
        dataPIL = dataPIL.resize(size=(self.img_width, self.img_height))
        # TODO 临时添加
        dataPIL.save(str(wave_index)+".png",directory='MNIST_data')
        plt.show()
        #
        # data = np.asarray(dataPIL)
        # data = data / 255
        # data = data[np.newaxis, :]
        # buffer_.close()
        # plt.close()

    def get_layout(self, layout_file: str) -> dict:
        """
        文件是md格式，其内容为|节点名称|节点横坐标|节点纵坐标|
        :param layout_file: 布局文件
        :return:
        """
        file = os.path.join('E:/graduate student/实验室/中兴-业务编排/test', layout_file)
        rtn = {}
        if os.path.isfile(file):
            datas = np.loadtxt(file, delimiter='|', skiprows=2, dtype=str)
            origin_data = datas[:, 1:(datas.shape[1] - 1)]
            for i in range(origin_data.shape[0]):
                rtn[origin_data[i][0]] = (float(origin_data[i][1]), float(origin_data[i][2]))
            return rtn
        else:
            raise FileNotFoundError

if __name__ == '__main__':
    a = RwaNetwork('9node_14link.md',5,True,None,'E:/graduate student/实验室/中兴-业务编排/test')
    print(a.edges)
    tt = PltRwa(224,224,55,1,0.05,'9node_14link.md')
    # tt.draw(a,'1','
    tt.draw1(a, '2', '5', '4', [('2', '5')], ['2','5'], None, 1)