## 对抗生成网络在控制层入侵的应用

> prepare for ECOC 2018

#### codes
clone from https://github.com/goodfeli/adversarial, which is the first work about GAN.

The translation of this paper is listed at http://blog.csdn.net/wspba/article/details/54577236


#### Generative Adversarial Networks
生成对抗网络，是机器学习中的一种生成模型算法，用于模拟真实数据的联合概率分布，以达到以假乱真的效果


参考论文列表：

* Reed, S., Akata, Z., Yan, X., Logeswaran, L., Schiele, B., & Lee, H. (2016). Generative Adversarial Text to Image Synthesis. Icml, 1060–1069. http://doi.org/10.1017/CBO9781107415324.004
* Yi, Z., Zhang, H., Tan, P., & Gong, M. (2017). DualGAN: Unsupervised Dual Learning for Image-to-Image Translation. Proceedings of the IEEE International Conference on Computer Vision, 2017–Octob, 2868–2876. http://doi.org/10.1109/ICCV.2017.310
* Che, T., Li, Y., Jacob, A. P., Bengio, Y., & Li, W. (2016). Mode Regularized Generative Adversarial Networks, 1–13. Retrieved from http://arxiv.org/abs/1612.02136
* Sønderby, C. K., Caballero, J., Theis, L., Shi, W., & Huszár, F. (2016). Amortised MAP Inference for Image Super-resolution, 1–17. http://doi.org/10.1007/s00138-014-0623-4
* Taigman, Y., Polyak, A., & Wolf, L. (2016). Unsupervised Cross-Domain Image Generation, 1–14. http://doi.org/10.1109/CVPR.2017.106
* Arjovsky, M., & Bottou, L. (2017). Towards Principled Methods for Training Generative Adversarial Networks, 1–17. http://doi.org/10.2507/daaam.scibook.2010.27
* Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. Retrieved from http://arxiv.org/abs/1701.07875
* Belghazi, M. I., Rajeswar, S., Mastropietro, O., Rostamzadeh, N., Mitrovic, J., & Courville, A. (2018). Hierarchical Adversarially Learned Inference, 1–18. Retrieved from http://arxiv.org/abs/1802.01071
* Donahue, J., Krähenbühl, P., & Darrell, T. (2016). Adversarial Feature Learning, (2016), 1–18. http://doi.org/10.1038/nphoton.2013.187
* Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based Generative Adversarial Network, (2014), 1–17. http://doi.org/10.1016/j.neunet.2014.10.001
* Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved Techniques for Training GANs, (Nips), 1–9. http://doi.org/arXiv:1504.01391
* Zhu, J. Y., Krähenbühl, P., Shechtman, E., & Efros, A. A. (2016). Generative visual manipulation on the natural image manifold. Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9909 LNCS, 597–613. http://doi.org/10.1007/978-3-319-46454-1_36
* Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved Training of Wasserstein GANs. http://doi.org/10.1016/j.aqpro.2013.07.003
* Denton, E., Chintala, S., Szlam, A., & Fergus, R. (2015). Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks, 1–9. Retrieved from http://arxiv.org/abs/1506.05751
* Villani, C. (2009). Optimal Transport: Old and New. Media, 338, 0,488. http://doi.org/10.1007/978-3-540-71050-9
* Wu, H., Zheng, S., Zhang, J., & Huang, K. (2017). GP-GAN: Towards Realistic High-Resolution Image Blending. Retrieved from http://arxiv.org/abs/1703.07195
* Kim, T., Cha, M., Kim, H., Lee, J. K., & Kim, J. (2017). Learning to Discover Cross-Domain Relations with Generative Adversarial Networks. Retrieved from http://arxiv.org/abs/1703.05192
* Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., … Bengio, Y. (2014). Generative Adversarial Nets. Advances in Neural Information Processing Systems 27, 2672–2680. http://doi.org/10.1017/CBO9781139058452
* Berthelot, D., Schumm, T., & Metz, L. (2017). BEGAN: Boundary Equilibrium Generative Adversarial Networks, 1–10. http://doi.org/1703.10717
* Zhang, H., Xu, T., Li, H., Zhang, S., Wang, X., Huang, X., & Metaxas, D. (2017). StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks, 5907–5915. http://doi.org/10.1109/ICCV.2017.629
* Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets, 1–7. Retrieved from http://arxiv.org/abs/1411.1784
* Montavon, G., Orr, G. G. B., Müller, K.-R., LeCun, Y., Bottou, L., Orr, G. G. B., … Müller, K.-R. (2012). Neural Networks: Tricks of the Trade. Springer Lecture Notes in Computer Sciences. http://doi.org/10.1007/3-540-49430-8
* Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2016). Image-to-Image Translation with Conditional Adversarial Networks, 1125–1134. http://doi.org/10.1109/CVPR.2017.632
* Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 1–16. http://doi.org/10.1051/0004-6361/201527329
* Nguyen, A., Clune, J., Bengio, Y., Dosovitskiy, A., & Yosinski, J. (2016). Plug & Play Generative Networks: Conditional Iterative Generation of Images in Latent Space, (1), 4467–4477. http://doi.org/10.1109/CVPR.2017.374

#### 核心想法

攻击者通过攻击运营商控制器，或者企业租赁专线/切片网络的控制器，可以获取其控制权限。
获得权限之后，有些攻击者为了持续窃取核心机密，抑或以非法控制的网络为伪装，进行一些违法活动，通常会在控制网络中潜伏下来。
在潜伏期间，如何将自己伪装成正常用户，以骗取越来越先进的异常检测技术，是一个难题。

控制器的响应简单分为主动请求和被动响应。主动请求中，需要伪装成正常请求以避免被发现。
```
(1). W. Fawaz, B. Daheb, O. Audouin, M. Du-Pond and G. Pujolle, "Service level agreement and provisioning in optical networks," in IEEE Communications Magazine, vol. 42, no. 1, pp. 36-43, Jan 2004.
doi: 10.1109/MCOM.2004.1262160
keywords: {multiprotocol label switching;optical fibre subscriber loops;telecommunication network management;telecommunication services;telecommunication traffic;GMPLS;SLA;bandwidth capacity;generalized multiprotocol label switching management;optical networks;optical service level agreement;provisioning process;service types;traffic;wavelength;Automation;Bandwidth;Intelligent networks;Multiprotocol label switching;Optical devices;Optical fiber networks;Proposals;Resource management;Telecommunication traffic;Web and internet services},
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1262160&isnumber=28217
```
主动业务请求的属性：

<center><bold>Table I<bold></center>

|Connection Setup Times|Premium|Gold|Silver|Bronze|
|:---|:---|:---|:---|:---|
|Leased line, preprovisioned bandwidth| 24 h| 4 days| 2 weeks| 2 months|
|Bandwidth on demand| 1 min|10 min|1 h|12h|

<center><bold>Table II<bold></center>

|Service Availability and resilience|Premium| Gold|Silver|Bronze|
|:----|:----|:----|:----|:----|
|Out-of-Service criterion| Degraded BER=$10^-4$|Degraded BER=$10^-3$|Falut(LOS)|Fault(LOS)|
|Recovery time with degraded SLA| Not specified| 50 ms| 500 ms| 5 s|
|Full recovery time| 50 ms| 300 ms| 5 s| 5 min|
|Service Unavailability| $10^-5$|$10^-4$|$10^-3$|$10^-2$|

<center><bold>Table III<bold></center>

|Service differentiation in routing|Premium| Gold|Silver|Bronze|
|:-----|:-----|:-----|:-----|:-----|
|Routing Stability| 2 times/year| 1 time/month| 1 time/week | No limition|
|Route Differentiation|Optional Fully supported(link, node, SRLG)|Optional Partially supported (link, node)|Optional Partially supported (link)| Not supported|
|Confidentiality Optional Optional|Fully supported Partially supported (O/E, grooming) (grooming)|Not supported |Not supported |


<center><bold>Table IV<bold></center>

|Performance guarantees|Premium| Gold|Silver|Bronze|
|:-----|:-----|:-----|:-----|:-----|
|Throughput |n × X Gb/s|n × X Gb/s |n × X Gb/s |n × X Gb/s |
|Maximum delay |35 ms |100 ms |500 ms |5 s|
|Jitter |3 ms |10 ms |50 ms |1 s|
|Packet loss |10^–9 |10^–6 |10^–4 |10^–2|