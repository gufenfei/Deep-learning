import torch.nn as nn
import torch

#残差模块类
class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self , in_channel, out_channel,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                               kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                              kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


#构建resNet类
class ResNet(nn.Module):
    def __init__(self,block,blocks_num,num_classes=1000,include_top=True):
        super(ResNet,self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3,self.in_channel,kernel_size=7,stride=2,
                               padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,blocks_num[0])
        self.layer2 = self._make_layer(block,128,blocks_num[1],stride=2)
        self.layer3 = self._make_layer(block,256,blocks_num[2],stride=2)
        self.layer4 = self._make_layer(block,512,blocks_num[3],stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*block.expansion,num_classes)
            
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    #构建残差网络模块
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)
    #前向传播函数
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.fc(x)
        return x

    #修改模型输出类的数量
    def resetOutputCalss(self,num_classes):
        in_chennel = self.fc.in_features
        self.fc =  nn.Linear(in_chennel,num_classes)
    

#构建res34网络
def resnet34(num_classes = 1000,include_top = True):
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)


#将模型装成onnx
def get_onnx_from_model():
    # net = resnet34()
    # print(net)
    # net.resetOutputCalss(5)
    # ##生成跨平台推理模型onnx
    # net.load_state_dict(torch.load("./backup/resNet34/resNet34_final.pth",map_location=None))
    # net.eval()
    # x = torch.randn(1,3,224,224,requires_grad=True)    #网络输入的大小
    # export_onnx_file = "./backup/resNet34/resNet34.onnx"	
    # torch.onnx.export(net,
    #                 x,
    #                 export_onnx_file,   #存储路径
    #                 input_names=["input"],    # 输入名
    #                 output_names=["output"])    # 输出名
    return

################################################
##以下为测试代码

def main():
    #get_onnx_from_model()
    
    net = resnet34()
    print(net)
    
    print('end')
    return 
    
if __name__ == "__main__":
    main()