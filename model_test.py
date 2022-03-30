import torch
from torchsummary import summary
import cv2
model = torch.load('./insurance_angle.pth')
test_img = "1307.jpg"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = cv2.imread(test_img)
img = cv2.resize(img, (224, 224))
img = cv2.dnn.blobFromImage(img, 1.0/127.5, (224,224), (127.5, 127.5, 127.5), False, False)
model.eval()
img = torch.from_numpy(img).to(device)
outputs = model(img)
print(torch.argmax(outputs))
#dummy_input1 = torch.randn(1, 3, 224, 224).to(device)
#summary(model,(3,224,224))
#print(model)
#input_names = [ "data"]
#output_names = [ "fc" ]
#torch.onnx._export(model,dummy_input1,"test.onnx",verbose=True,export_params=True,input_names=input_names,output_names=output_names, keep_initializers_as_inputs=True, opset_version=11)


