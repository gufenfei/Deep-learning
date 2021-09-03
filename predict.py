import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from model import resnet34
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

img = Image.open('./data/flower_data/val/daisy/3337536080_1db19964fe.jpg')
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img , dim=0)

try:
    json_file = open('./data/flower_data/class_indices.json','r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = resnet34(num_classes = 5)
model_weight_path = './backup/resNet34/resNet34_final.pth'
model.load_state_dict(torch.load(model_weight_path))
model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output,dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)],predict[predict_cla].numpy())
plt.show()

