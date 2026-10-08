# Stage 2：Patch extraction PASS

新增patch_extractor.py/test_extraction.py，固定P=200/S=100，不搜索。976 starts=[0,100,200,300,400,500,600,700,776]共9；2000 starts=0..1800每100共19。first/last内容、尾覆盖、去重、不越界、determinism PASS。0/1/199无完整patch，200一个，201带tail两个，300两个；不从短序列补造样本。Mixed lengths976/2000/199/201 pad到19，只index valid patch positions，不读true_length后samples。改变sample padding为NaN不影响extraction；invalid patches为0；autograd无padding梯度，异常length拒绝。

无模型训练，无outer，无recipe更改，无protocol偏离；结果仅correctness。下一步允许P1 Bag-of-Patches实现及测试，不按分数选设计。
