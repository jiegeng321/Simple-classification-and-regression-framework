import torch
#from torchsummary import summary
#model = torch.load('/home/admin/data/xfeng/COMMODEL/face_pts_20th_with_init_myself/best_acc.pth')
model = torch.load('/home/admin/data/xfeng/COMMODEL/face_pts_20th_with_init_final_no_open_small2/best_acc.pth')
#model = torch.load('/home/admin/data/xfeng/COMMODEL/face_pts_20th_with_init_final_no_open/best_acc.pth')
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dummy_input1 = torch.randn(1, 3, 144, 144).to(device)
#summary(model,(3,224,224))
#print(model)
input_names = [ "input"]
output_names = [ "output" ]
torch.onnx._export(model,dummy_input1,"face_pts_small_20210220.onnx",verbose=False,export_params=True,input_names=input_names,output_names=output_names, keep_initializers_as_inputs=True, opset_version=11)


