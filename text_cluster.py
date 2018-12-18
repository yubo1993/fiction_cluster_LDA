# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 09:41:01 2018

@author: YUBO
"""

import os
import pandas as pd
import jieba  
import nltk
from matplotlib.font_manager import FontProperties
from sklearn.datasets.base import Bunch
import matplotlib.pyplot as plt
import pickle#持久化类
from sklearn.feature_extraction.text import TfidfVectorizer#TF-IDF向量生成类
#找小说集的文件夹名称
corpus_path1="D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\Fiction_collection\\"

fiction_collection=os.listdir(corpus_path1)


def readfile(path):
    fp = open(path, "r", encoding='utf-8', errors='ignore')
    content = fp.read()
    fp.close()
    return content
def savefile(savepath, content):
    fp = open(savepath, "w",encoding='utf-8', errors='ignore')
    fp.write(content)
    fp.close()
seg_path='D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\'
#找到小说集每个文件夹中的txt文件
for mydir in fiction_collection:
    class_path = corpus_path1 + mydir + "\\"  # 拼出分类子目录的路径
    seg_dir = seg_path + mydir + "\\"  # 拼出分词后预料分类目录
    if not os.path.exists(seg_dir):  # 是否存在，不存在则创建
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)
    for file_path in file_list:
        fullname = class_path + file_path
        content = readfile(fullname).strip()  # 读取文件内容
        content = content.replace("\r\n", "").strip()  # 删除换行和多余的空格
        ## 添加自定义词典定义一些地名和人名，使分词更准确
        jieba.load_userdict("D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\names.txt")
        content_seg = jieba.cut(content)
        savefile(seg_dir + file_path, " ".join(content_seg))

print("分词结束")

#对分词后的结果进行处理

"""
Bunch 类提供了一种key，value的对象形式
target_name 所有分类集的名称列表
label 每个文件的分类标签列表
filenames 文件路径
contents 分词后文件词向量形式
"""
def corpus2Bunch(wordbag_path,seg_path):  
    catelist = os.listdir(seg_path)# 获取seg_path下的所有子目录，也就是分类信息  
    #创建一个Bunch实例  
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])  
    bunch.target_name.extend(catelist)    
    # 获取每个目录下所有的文件  
    for mydir in catelist:  
        class_path = seg_path + mydir + "\\"  # 拼出分类子目录的路径  
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件  
        for file_path in file_list:  # 遍历类别目录下文件  
            fullname = class_path + file_path  # 拼出文件名全路径  
            bunch.label.append(mydir)  
            bunch.filenames.append(fullname)  
            bunch.contents.append(readfile(fullname))  # 读取文件内容  
    # 将bunch存储到wordbag_path路径中  
    with open(wordbag_path, 'wb') as file_obj:  
        pickle.dump(bunch, file_obj)  
    print("构建文本对象结束！！！")


if __name__ == '__main__': 
    #对文本进行Bunch化操作：  
    wordbag_path = 'D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\train_set.dat'  # Bunch存储路径  
    seg_path = 'D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\'  # 分词后分类语料库路径  
    corpus2Bunch(wordbag_path, seg_path)  
 
    

# 读取bunch对象  
def readbunchobj(path):  
    with open(path, 'rb') as file_obj:  
        bunch = pickle.load(file_obj)  
    return bunch  
  
# 写入bunch对象  
def writebunchobj(path, bunchobj):  
    with open(path, 'wb') as file_obj:  
        pickle.dump(bunchobj, file_obj)  
"""
停止词空间构造，由于小说中包含很多人名会占很大频率,但人名不在我们的研究
范围内，所以考虑将常见的人名也当做停止词

filenames=os.listdir("D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\stop_words")
#打开当前目录下的result.txt文件，如果没有则创建
f=open("D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\stop_words\\stop_words.txt",'w')
#先遍历文件名
for filename in filenames:
    filepath = "D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\stop_words"+'/'+filename
    #遍历单个文件，读取行数
    content = readfile(filepath).strip()  # 读取文件内容
    content = content.replace("\r\n", "").strip()  # 删除换行和多余的空格
    for line in content:
        f.writelines(line)
#关闭文件
f.close()
"""
#这个函数用于创建TF-IDF词向量空间  
def vector_space(stopword_path,bunch_path,space_path):    
    stpwrdlst = readfile(stopword_path).splitlines()#读取停用词  
    bunch = readbunchobj(bunch_path)#导入分词后的词向量bunch对象  
    #构建tf-idf词向量空间对象  
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})  
    vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True,max_df=0.8, min_df=0.2,use_idf=True)    
    #此时tdm里面存储的就是if-idf权值矩阵  
    tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)  
    tfidfspace.vocabulary = vectorizer.vocabulary_  
    writebunchobj(space_path, tfidfspace)  
    print ('tf-idf词向量空间实例创建成功！！！')
  
if __name__ == "__main__":  
    stopword_path = 'D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\stop_words.txt'#停用词表的路径  
    bunch_path = 'D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\train_set.dat'  #导入Bunch的路径  
    space_path = 'D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\tfdif_space.dat'  # 词向量空间保存路径  
    vector_space(stopword_path,bunch_path,space_path)  

#接下来是文本聚类
trainpath = 'D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\tfdif_space.dat' 
train_set = readbunchobj(trainpath)  
from sklearn.cluster import KMeans
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(train_set.tdm)
clusters = km.labels_.tolist()
cluster_out=pd.DataFrame(data={'fiction_name':train_set.filenames,'fiction_author':train_set.label,
                               'fiction_cluster':clusters})
## 查看每类有多少个
count1 = cluster_out.groupby("fiction_cluster").count()

## 可视化
count1 = count1.reset_index()
count1.plot(kind="barh", figsize=(6,5), x="fiction_cluster", y="fiction_author", legend=False)
for xx,yy,s in zip(count1.fiction_cluster,count1.fiction_name,count1.fiction_author):
  plt.text(y =xx-0.1, x = yy+0.5,s=s)
plt.ylabel("cluster label")
plt.xlabel("number")
plt.show()

#分别查看每个作者的作品的类别情况，并看与其他作者的交叉情况
count2 = cluster_out.groupby("fiction_author").agg({'fiction_cluster':'unique'})
count3=cluster_out.groupby(["fiction_author",'fiction_cluster']).agg({'fiction_name':'count'})
count1.to_csv("D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\count1.csv",index=False)
count2.to_csv("D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\count2.csv",index=False)
count3.to_csv("D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\count3.csv",index=False)
cluster_out.to_csv("D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\cluster_out.csv",index=False)



#接下来分别查看没类作品中的关键词进行主题抽取和词云绘制（每类中只选取一部作品进行分析）
"""
【0】代表人物是：诸葛青云大部分作品
选取的代表作是《一剑光寒十四州》
【1】代表人物有：古龙的部分作品，小椴，李凉，柳残阳，王晴川，萧逸以及诸葛青云的部分作品
选取代表作是古龙的《苍穹神剑》 萧逸《剑仙传奇》
【2】代表人物：金庸，梁羽生
代表作选择金庸《侠客行》梁羽生《七剑下天山》
【3】代表人物：步非烟，沧月
代表作选择步非烟的《剑侠奇缘》
【4】代表人物：古龙
选择的代表作是《楚留香2--蝙蝠传奇》
"""
shisizhou_path="D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\诸葛青云武侠小说集\\一剑光寒十四州-诸葛青云_TXT小说天堂.txt"
cangqiongshenjian_path="D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\古龙小说集\\cangqiongshenjian_gulong.txt"
jianxianchuanqi_path="D:\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\萧逸武侠小说集\\剑仙传奇-萧逸_TXT小说天堂.txt"
xiakexing_path="D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\金庸小说集\\xiakexing_jinyong.txt"
qijian_path="D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\梁羽生全集\\七剑下天山.txt"
jianxia_path="D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\步非烟作品集\\剑侠情缘.txt"
chuliuxiang2_path="D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\word\\古龙小说集\\chuliuxiangxinchuan2_bianfuchuanqi_gulong.txt"

#绘制词云的函数

from wordcloud import WordCloud
def word_cloud(path):
    #读取标点符号库
    text = readfile(path).strip()  # 读取文件内容
    text = text.replace("\r\n", "").strip()  # 删除换行和多余的空格
    f=open('D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\stop_words.txt',"r",encoding='utf-8', errors='ignore')
    stopwords={}.fromkeys(f.read().split("\n"))
    f.close()
    segs=jieba.cut(text,cut_all=True)
    mytext_list=[]
    #文本清洗
    for seg in segs:
        if seg not in stopwords and seg!=" " and len(seg)!=1:
            mytext_list.append(seg.replace(" ",""))
    cloud_text="//".join(mytext_list)
    
    wc = WordCloud(
            # 设置背景颜色
            background_color="black",
            # 设置最大显示的词云数
            max_words=50,
            font_path='C:\Windows\Fonts\simfang.ttf',
            height=2000,
            width= 3000,
            # 设置字体最大值
            max_font_size=600,
            # 设置有多少种随机生成状态，即有多少种配色方案
            random_state=20
            )
 
    myword = wc.generate(cloud_text)  # 生成词云
    # 展示词云图
    plt.imshow(myword)
    plt.axis("off")
    plt.show()

if __name__=="__main__":
    word_cloud(shisizhou_path)
    word_cloud(cangqiongshenjian_path)
    word_cloud(jianxianchuanqi_path)
    word_cloud(xiakexing_path)
    word_cloud(qijian_path)
    word_cloud(jianxia_path)
    word_cloud(chuliuxiang2_path)
    
#主题模型
    

from sklearn.feature_extraction.text import CountVectorizer    
res1=readfile(shisizhou_path).strip()
res2=readfile(cangqiongshenjian_path).strip()
res3=readfile(jianxianchuanqi_path).strip()
res4=readfile(xiakexing_path).strip()
res5=readfile(qijian_path).strip()
res6=readfile(jianxia_path).strip()
res7=readfile(chuliuxiang2_path).strip()
corpus=[res1,res2,res3,res4,res5,res6,res7]
f=open('D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\stop_words.txt',"r",encoding='utf-8', errors='ignore')
stopwords={}.fromkeys(f.read().split("\n"))
f.close()
cntVector = CountVectorizer(stop_words=stopwords,max_features=200,ngram_range=(1,3),max_df=0.7,min_df=0.3)
cntTf = cntVector.fit_transform(corpus)
feature_names=cntVector.get_feature_names()
print(cntTf  )
import lda
lda_topic = lda.LDA(n_topics=10,random_state=41)
docres=lda_topic.fit_transform(cntTf)

print(docres)
print(lda_topic.components_)


import seaborn as sns
def plot_topic(doc_topic):
    f, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(doc_topic, cmap=cmap, linewidths=0.05, ax=ax)
    ax.set_title('proportion per topic in every fuction')
    ax.set_xlabel('topic')
    ax.set_ylabel('fiction')
    plt.show()

    f.savefig('D:\\数据分析精选大礼包\\【13】nlp\\text_mining\\topic_heatmap.jpg', bbox_inches='tight')

if __name__=="__main__":
    plot_topic(lda_topic.doc_topic_)








