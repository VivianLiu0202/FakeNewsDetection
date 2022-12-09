# FakeNewsDetection
Python大作业虚假新闻检测
# 一、问题描述
数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在train.news.csv，测试数据保存在test.news.csv。
实验过程中先统计分析训练数据【train.news.csv】。根据train.news.csv中的Title文字训练模型，然后在test.news.csv上测试，给出Precision, Recall, F1-Score, AUC的结果。

# 二、开发环境
Python Interpreter：Python 3.9
Python IDE:PyCharm CE

# 三、数据集说明
数据集包含微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label（0为真1为假）。训练时将Official Account Name，Title，Report Content三列数据合成一列。而News Url，Image Url则舍去不用。
我们观察训练集的时候可以发现，训练集的真消息全部分布在前面，假消息全部分布在后面，因此我们在进行模型训练的时候需要把数据shuffle一下，避免模型根据数据的先后位置进行学习判断。

# 四、关键步骤及代码细节
（1）Prepraing Training Data
首先读入数据，并编写cleaning函数清洗数据并使用tokenizer提取关键词生成字典。此后将序列转换为经过填充之后的一个长度相同的新序列，即数据集处理完成。

（2）Model Training
  建立序贯模型(Sequential)。在Keras中，序贯模型是单输入单输出，层与层之间只有相邻关系，没有跨层的连接。这种模型的编译速度较快，也可以清晰地表明各层在数据输入到输出的转换过程中起到的职责。
  模型的第一层是嵌入层(Embedding)，它使用长度为128的向量来表示每一个词语，嵌入层将词语转换为词向量，使用词嵌入的目的是在词之间找到更好的相似性，较独热编码更好。
  模型的第二层是LSTM层，使用Tensorflow的双向LSTM模型进行堆叠。第一个双向层定义为64个单元，第二层定义为32个单元。通过多次堆叠发现，当堆叠层数为1或2时report中的ACC数值较高，并且训练的时间较短。
  之后我们使用具有64个单元的Dense层和激活函数relu
  最后一层是具有sigmoid逻辑激活函数的Dense层，由于是二分类问题，我们在输出层仅设置一个神经元。
  同时在此之间设置Dropout层，Dropout是作为缓解过拟合而被提出的一种正则化方法，它确实能够有效缓解过拟合现象的发生，但是Dropout带来的缺点就是可能会减缓模型收敛的速度，因为每次迭代只有一部分参数更新，可能导致梯度下降变慢。将Dropout的数值设置为0.2，即每次使20%的神经元失活。
  之后就是对模型进行编译。使用adam优化函数进行反向传播。并使用二元对数交叉熵损失函数binary_crossentropy进行度量的损失和准确性。损失函数用于优化模型，而度量用于我们的比较。
接下来就是模型的训练。我设置训练轮数epochs=5，进行梯度下降时每个batch包含的样本数batch_size为32。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。

  在训练完成之后使用matplotlib进行绘图，将5轮训练和测试的acuuracy和loss通过绘制图像表示。
（3）Testing Model on Testing Dataset
  最后将测试集数据进行清洗、序列预处理后进行预测，并输出分类报告，AUC等实验需要数据的值。

# 五、结果分析与模型改进
Recall：实际为x的类别中，有多少预测为x
Precision：预测为x的样本中，有多少被正确预测为x
F1-score = 2 * precision * recall /(precision+recall)。
macro avg = 上面类别各分数的直接平均
weighted avg = 上面类别各分数的加权（权值为support）平均
ACC:全部样本里被分类正确的比例
PRC:在正负样本分布得极不均匀(highly skewed datasets)，负例远大于正例时，并且这正是该问题正常的样本分布时，PRC比ROC能更有效地反应分类器的好坏，即PRC曲线在正负样本比例悬殊较大时更能反映分类的真实性能。

  观察实验数据的report我们发现虽然该模型的ACC有91%，看起来还行，但是1类别中的Precision和Recall都偏低。特别是Recall实际为1的类别中仅有一半能被预测为1，说明该模型识别假信息的能力还有待改进。预计原因是由于训练集中的真消息过多假消息较少，正负样本分布严重不均匀。因此模型的训练程度不够。
  但由于该模型的主要功能为虚假新闻检测，因此对于Recall的要求较高，因此应当在保证基本的模型精确度的情况下尽可能地提高召回率，保证对于虚假信息的检测更加准确。比如可以通过多堆叠其他层，或者使用更加好的模型的方式进行改进。
  观察Precision-Recall Curves我们发现在当需要获得更高Recall时，model需要输出更多的样本，Precision可能会伴随出现下降/不变/升高，得到的曲线会出现浮动差异（出现锯齿），无法像ROC一样保证单调性。所以，对于正负样本分布大致均匀的问题，ROC曲线作为性能指标更棒。但是由于本题的正负样本分布非常不均匀，所以PRC比ROC能更有效地反应分类器的好坏。


# 六、总结反思
  本次大作业是我第一次接触到机器学习和深度学习的相关知识。虽然在最开始做的时候我几乎毫无头绪，但是通过在网上学习各种模型我逐渐有了思路。在写模型的时候我对于深度学习的框架有了一定的认识，并且学到了很多库的用法。
  但是，我的模型仍然存在一些问题：
1.模型准确率仍不够高，可以通过多堆叠几层，或者使用更加好的模型；
2.我使用的是tensorflow的cpu版本，训练模型时间较长，可以改用gpu版本（但是不太会配置x）；
3.最后将预测结果转化为01的序列时可以采用one_hot编码，在Dense层使用softmax函数，使用两个神经元。但是不知道什么原因我的电脑无法使用独热编码，因此只能简单的写一个if-else语句来进行划分；
4.report中的数据也不太满意。Recall和F1-score比较低。
  希望之后的学习可以让我更加了解关于深度学习的东西。

