## 本工程旨在探索深度学习方法在光网络中的应用（Python实现）

### 1、1channel_access_check

利用graphviz工具生成1通道拓扑，进行路由可达性判断。训练集和测试集可见百度网盘同名目录。
根据日志文件可知，在resnet18网络中，可以达到96.605%的判断准确率。


### 2. 40channel_access_check

利用graphviz工具生成40通道拓扑，进行路由可达性判断。训练集和测试集可见百度网盘同名目录。
由于一个样例需要40张图片来组成，体积之大已经使得实验的进行不可能，因此搁置。

### 3. DL4FlappyBird

从github上爬下来的一个FlappyBird的游戏，DQN算法是用pytorch实现的。跑了两天的程序，最后最高纪录能玩到20下左右。
程序本身应该有问题，我感觉主要是一次性拿到前后4帧的画面，但是真实情况下只拿到1帧，把这一帧重复了四次做的训练。总之不是很成功。


### 4. gymPlatform

为了熟悉openai出品的gym和universe。发现universe的API虽然大部分兼容gym，但是并不好用。
其次，universe的环境真的不好配，还有vnc等配套可视化的要求，不容易在集群上使用。因此，
后来我果断放弃了universe，转而只看gym。

#### I. CartPoleDemo

用了A2C算法，游戏详情可见该项目目录下的readme。

#### II. FlappyBird-v0

gym中的Flappybird，还没有实现。

### 5. GRWA

#### I. 数据集生成

数据集生成的部分可见data_gen目录