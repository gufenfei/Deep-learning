import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
import numpy as np
import math
import json    #用于操作json文件

from model import resnet34


#将训练和测试可能使用到的参数封装成一个类
class config():
    #输入 
    INPUT_WIDTH = 224
    INPUT_HEIGHT = 224
    INPUT_CHANNELS = 3
    DATA_PATH = './data/flower_data'
	TRAIN_DATA_PATH = config.DATA_PATH+'/train'
	TEST_DATA_PATH = config.DATA_PATH+'/test'
    NUM_IMAGE = -1 #表示训练的训练集的总图像数量
    def setNumberImage(dataset):    #传入数据集，计算图片数量，并存入
        config.NUM_IMAGE = len(dataset)
    
    #输出
    OUTPUT_CLSSSES = 5
    
    #训练
    BATCH_SIZE = 32
    MAX_EPOCH = 50
    LEARNING_RATE = 0.00001
    STEP_EPOCH = [10,20,30]
    SCALES_LEARNING = [0.3,0.1,0.1]
    MOMENTUM = 0.9
    DECAY = 0.0005

    #模型保存
    SAVE_EACH_EPOCH = 5
    NAME_MODEL = 'resNet34'

    #测试
    BATCH_SIZE_TEST = 1



#############################################################
#一个可以用来绘制LOSS曲线的类
#用来观察训练过程中的loss变化，直观体现当前模型的拟合效果
class Chart():
    def __init__(self,max_epoch,size_batch,num_image,max_y = 0.5,linewidth = None,linecolor = None,linestyle = None,lineMarker = None,
                 fontsize = None,labelcolor = None,labeloffset = None,str_labelX = '',str_labelY = ''):
        
        self.max_epoch = max_epoch
        self.num_image = num_image
        self.size_batch = size_batch
        self.max_batch = math.ceil(self.num_image/self.size_batch)
        
        self.current_x = 0
        self.current_y = 0
        self.value_y = []    #用来记录所有输入的y值
        
        self.max_x = self.max_batch * self.max_epoch
        self.max_y = max_y
        
        self.__coordinate_x = []   # 外部不可访问，主要是存储输入的折线图坐标信息，这里的值不能代表真实值
        self.__coordinate_y = []
        
        self.__gridPoint_x = []    # 储存坐标系的网格点信息
        self.__gridPoint_y = []
        
        self.__initGrad()#这里会计算出所有的网格的信息，会将一部分参数增加到类的成员中
        self.setLineProperties(linewidth,linecolor,linestyle,lineMarker)
        self.setLabelProperties(fontsize,labelcolor,labeloffset)
        self.setLabelX(str_labelX)
        self.setLabelY(str_labelY)
        #print(self.max)
        return
    
    #绘制曲线的函数，参数：当前的y值  和  自定义的输出信息
    def dispChart(self,current_y,text_user = ''):
        #plt.ioff()
        plt.clf()
        plt.ion()
        #plt.figure()
        
        #设置坐标系的参数
        #print(self.max_x)
        plt.xlim(0,self.max_x)
        plt.ylim(0,self.max_y)
        #print(self.max_x/self.num_epoch)
        ##设置X轴的网格和标签
        plt.xticks(self.__ticks_x,self.__txt_x)
        plt.xlabel(self.__label_x)
        ##设置Y轴的网格和刻度
        plt.yticks(self.__ticks_y,self.__txt_y)
        plt.ylabel(self.__label_y) 
        plt.grid(axis='both') #设置完毕，显示网格
        
        #设置绘图坐标和绘制loss线条
        self.__coordinate_x.append(self.current_x)
        self.current_y = current_y
        self.value_y.append(self.current_y)
        self.__coordinate_y.append(min(self.current_y,self.max_y*0.999))#保证绘制在图内
        #print(min(self.current_y,self.max_y))
        plt.plot(self.__coordinate_x,self.__coordinate_y,
                 linewidth = self.__linewidth,color = self.__linecolor,linestyle = self.__linestyle,marker = self.__lineMarker)
        
        #显示相关的信息
        index_epoch = int(self.current_x/self.max_batch) + 1
        index_batch = self.current_x%self.max_batch + 1
        self.current_x+=1
        plt.text(0,self.max_y*self.__offset,'epoch: %d, batch: %d, %s: %f        %s'%(index_epoch,index_batch,self.__label_y,self.current_y,text_user),
                 fontsize = self.__fontsize,color = self.__labelColor)

        #保存图像
        plt.pause(0.0001)
        plt.savefig('./chart.png')
        return 
    
    #设置曲线相关的属性
    def setLineProperties(self,linewidth = None,color = None,linestyle = None,lineMarker = None):
        self.__linewidth = 2 if linewidth == None else linewidth
        self.__linecolor = (0,0.5,0) if color == None else color
        self.__linestyle = None if linestyle == None else linestyle
        self.__lineMarker = None if lineMarker== None else lineMarker
        return 
    
    #设置相关信息字符的属性
    def setLabelProperties(self,fontsize = None,color = None,offset = None):
        self.__fontsize = 12 if fontsize == None else fontsize
        self.__labelColor = (0,0,0) if color == None else color
        self.__offset = 1.01 if offset == None else offset + 1
        return 

    def setLabelY(self,str):
        self.__label_y = 'loss' if str=='' else str
        
    def setLabelX(self,str):
        self.__label_x = 'epoch' if str=='' else str
    
    def getAllValue(self):
        return self.value_y
    
    def __initGrad(self):
        #设置x的标签和网格点
        stride_epoch = 1 if self.max_epoch <= 10 else self.max_epoch/10
        self.__txt_x = [i*stride_epoch for i in range(min(self.max_epoch,10))]
        self.__ticks_x = [i*self.max_batch for i in self.__txt_x]
        # print(txt_x)
        # print(ticks_x)
        #设置y的标签和网格点
        self.__ticks_y = np.linspace(0,self.max_y,26)
        self.__txt_y = []
        for i in range(len(self.__ticks_y)):
            self.__txt_y.append(self.__ticks_y[i] if i%5==0 else '')
        return

'''
初始化中各种类型控制的控制字
1.线条颜色 除下表之外，也可以使用16进制的RGB值来设置线条颜色例 如 '#FF0000'红色
    =========    ===============================
    character    color
    =========    ===============================
    'b'          blue 蓝
    'g'          green 绿
    'r'          red 红
    'c'          cyan 蓝绿
    'm'          magenta 洋红
    'y'          yellow 黄
    'k'          black 黑
    'w'          white 白
    =========    ===============================

2.线条类型 linestyle
    =========    ===============================
    character    description
    =========    ===============================
    '-'          solid line style 实线
    '--'         dashed line style 虚线
    '-.'         dash-dot line style 点画线
    ':'          dotted line style 点线
    =========    ===============================

3.点型控制  lineMarker 
    =========    ===============================
    character    description
    =========    ===============================
    '.'          point marker 小圆点
    ','          pixel marker 像素点  's'
    'o'          circle marker 大圆点
    'v'          triangle_down marker 下三角形
    '^'          triangle_up marker 上三角形
    '<'          triangle_left marker 左三角形
    '>'          triangle_right marker 右三角形
    '1'          tri_down marker 下三叉
    '2'          tri_up marker 上三叉
    '3'          tri_left marker 左三叉
    '4'          tri_right marker 右三叉
    's'          square marker 正方形
    'p'          pentagon marker 五角星
    '*'          star marker 星号/乘号
    'h'          hexagon1 marker 六边形1
    'H'          hexagon2 marker 六边形2
    '+'          plus marker 加号
    'x'          x marker X型
    'D'          diamond marker 菱形
    'd'          thin_diamond marker 瘦菱形
    '|'          vline marker 短竖线
    '_'          hline marker 短横线
    =========    ===============================
'''
###################################################################
#训练网络
def train():
    print('train start')
    
    net = resnet34()
    
    #载入预训练模型
    model_weight_path = './data/resnet34-pre.pth'
    missing_keys,unexpected_keys = net.load_state_dict(torch.load(model_weight_path),strict=False)
    
    #重置全连接层最后一层
    net.resetOutputCalss(config.OUTPUT_CLSSSES)
    print(net)
    net = net.cuda()    #将模型移动到显存中，使用GPU训练
    
    #准备数据
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH,transform = transform)    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.BATCH_SIZE,
                                           shuffle=True,num_workers=0)#仅使用主线程载入数据，不使用多线程
    config.setNumberImage(train_dataset)#将数据集图像总数更新到训练参数中
    
    #将从数据集中读到的类名（文件夹名)保存为json文件
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val,key) for key,val in flower_list.items())
    json_str = json.dumps(cla_dict,indent=4)
    with open('./'+config.DATA_PATH+'/class_indices.json','w') as json_file:
        json_file.write(json_str)
    
    #损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimzer = optim.Adam(net.parameters(),lr= config.LEARNING_RATE)
    
    #初始化loss曲线图
    loss_chart = Chart(config.MAX_EPOCH,config.BATCH_SIZE,config.NUM_IMAGE,max_y = 0.5,linewidth=1,lineMarker='.')
    
    text_lr = config.LEARNING_RATE
    
    for epoch in range(config.MAX_EPOCH):
        for  i,data in enumerate(train_loader,start=0):
            optimzer.zero_grad()    #梯度清零
            images,labels = data    #获取数据和标签
            
            images = images.cuda()    #将图像移动到显存中
            labels = labels.cuda()    #将标签移动到显存中
            
            logits = net(images)    #前向传播
            
            loss = loss_function(logits,labels)    #计算loss
            loss.backward()    #反向传播
            optimizer.step()    #更新模型参数
            
            #可视化部分
            loss_cpu =  loss.cpu()
            loss_chart.dispChart(loss_cpu,'lr:%f'%text_lr)    #绘制曲线图
            print("epoch : %d, batch : %d, loss : %f     %f" % (epoch+1,i+1,loss_cpu,text_lr))
        #模型保存
        # if epoch % config.SAVE_EACH_EPOCH == config.SAVE_EACH_EPOCH-1:
        #     torch.save(model.state_dict(),"./backup/%s/%s_%d.pth"%(config.NAME_MODEL,config.NAME_MODEL,epoch+1))
        #     torch.save(model.state_dict(),"./backup/%s/%s_last.pth"%(config.NAME_MODEL,config.NAME_MODEL))           
        #学习率衰减
        if(epoch in config.STEP_EPOCH):
            for param in optimizer.param_groups:
                param['lr'] *= config.SCALES_LEARNING[config.STEP_EPOCH.index(epoch)]
                text_lr = param['lr']
    torch.save(model.state_dict(),"./backup/%s/%s_final.pth"%(config.NAME_MODEL,config.NAME_MODEL))    #最终结果保存一次
    return


####################################################################
##########                 以下为测试部分                 ##########
####################################################################


#训练参数保存模块测试函数
#测试参数模块对否能正常读写
def testConfig():
    print("config params test...")
    print(config.NUM_IMAGE)
    config.NUM_IMAGE = 1000
    print(config.NUM_IMAGE)


#LOSS曲线绘制类测试函数
#运行用来测试绘制类是否正常工作
def testClassChart():
    print("Line chart test...")
    
    config.MAX_EPOCH = 100
    config.NUM_IMAGE = 40
    config.BATCH_SIZE = 32
    loss_chart = Chart(config.MAX_EPOCH,config.BATCH_SIZE,config.NUM_IMAGE,max_y = 0.5,linewidth=1,lineMarker='.')
    loss = 0.55
    for i in range(loss_chart.max_x):
        loss*=0.995
        #print(a)
        loss_chart.dispChart(loss)
    return 

####################################################################

def main():
    #testConfig()
    #testClassChart()

    train()
    
    print("end")
    return

if __name__=="__main__":
    main()
