from torchsummary import summary

from train import create_model

input_size = 256
model = create_model(type="resnet34", input_size=input_size, num_classes=28)

summary(model, input_size=(4, input_size, input_size))
