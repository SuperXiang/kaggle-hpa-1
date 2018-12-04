from torchsummary import summary

from train import create_model

input_size = 256
model = create_model(type="alexnet", input_size=input_size, num_classes=340)

summary(model, input_size=(3, input_size, input_size))
