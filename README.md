# 议论文篇章结构评分系统

# 一、项目概述

本项目为自动作文评分中篇章结构的评分，第一版本方案（篇章结构评分系统V1.0.0）主要包括篇章结构特征提取模块、篇章结构评分模型训练模块以及模型测试模块三部分，后期根据模型效果再加以优化改进。

# 二、功能接口定义

输入数据：一篇文章
输出数据：该文章的篇章结构分数

# 三、总体流程

目前的方案主要是判断篇章结构的完整度，一个完整的议论文文章应该包含以下篇章结构：

## 3.1 开头段

* 背景（background）：\<BG>\</BG>
* 观点（point）：\<PT>\</PT>
* 过渡句（transition）：\<TST>\</TST>

## 3.2 理由段

* 理由（reason）：\<RS>\</RS>
* 理由解释（reason explanation）：\<REXP>\</REXP>
* 例子（example）：\<EG>\</EG>
* 例子解释（example explanation）：\<EEXP>\</EEXP>
* 例子泛化（generalization）：\<GRL>\</GRL>

## 3.3 让步段

* 承认对方优点（admission）：\<ADM>\</ADM>
* 反驳对方观点（retort）：\<RTT>\</RTT>

## 3.4 总结段

* 理由总结（sum reason）：\<SRS>\</SRS>
* 观点重申（reaffirm）：\<RAFM>\</RAFM>

## 3.5 无关段
（与文章无关的句子，有没有这部分都不影响篇章结构分数）

* 与文章无关（irrelevant）：\<IRL>\</IRL>

我们使用篇章结构评分系统来对测试集中每篇文章中的每个句子进行篇章结构类别分类，以篇章结构是否完整为评价标准进行作文评分。

篇章结构评分系统思维导图见image目录下的“篇章结构评分系统思维导图.png”

## 3.6 特征提取

### 3.6.1 对于结构特征：

* 利用NLTK对句子进行词分割，进而统计句子中词的个数;
* 以段落的句子结束的标点符号进行分割句子，位置特征由两部分组成：句子所在段落位置和句子在段落中的位置；
* 标点符号特征分为三类：句号、问号和感叹号。

### 3.6.2 对于词汇特征：

* 使用SRILM中的n-gram工具来训练语言模型，分别对不同类别的篇章结构进行训练，构建n-gram训练模型，然后再计算某个句子属于不同类别篇章结构的概率，将这个概率作为一个数值特征；
* 使用NLTK中的POS工具对句子中的单词进行词性标注。

### 3.6.3 对于句法特征：

* 使用NLTK中的Stanford分析器构建分析树（分析树图见image目录下的“分析树图.png”），我们提取分析树的深度作为一个数值特征，分析树如下图3所示；
* 我们主要以一个句子中的动词和谓语的时态作为句子的时态，而动词和谓语的时态识别可以由词性标注步骤来完成。

### 3.6.4 对于指示词特征：

* 将一个句子中的动词、副词、情态动词以及第一人称代词与事先归纳的指示词表进行对比，统计一个指示词与所属类别之间的概率统计。

# 四、参考文献

1.Christian Stab, Iryna Gurevych. Identifying Argumentative Discourse Structures in Persuasive Essays, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014, pages 46-56.

2.Rashmi Prasad, Eleni Miltsakaki, Nikhil Dinesh, Alan Lee, Aravind Joshi, Livio Robaldo, and Bonnie L. Webber. 2007. The Penn Discourse Treebank 2.0 annotation manual. Technical report, Institute for Research in Cognitive Science, University of Pennsylvania.

3.Roger Levy.Working with n-grams in SRILM.2015

4.http://www.nltk.org/api/nltk.html

5.https://ynuwm.github.io/2017/05/24/SRILM%E8%AE%AD%E7%BB%83%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%AE%9E%E6%88%98/