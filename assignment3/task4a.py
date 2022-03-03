import torchvision
from torch import nn
from trainer import Trainer, compute_loss_and_accuracy
import utils
from dataloaders import load_cifar10
from task2 import get_final_result, plot_training_validation_loss


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
                                           # as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected layer
            param.requires_grad = True 
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional layers
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size,
                               resolution=(224, 224),
                               mean=(0.485, 0.456, 0.406),
                               std=(0.229, 0.224, 0.225))

    model = Model()
    trainer = Trainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stop_count=early_stop_count,
        epochs=epochs,
        model=model,
        dataloaders=dataloaders,
        optimizer='Adam '
    )
    trainer.train()

    plot_training_validation_loss(trainer, '4a_train_val_loss')
    get_final_result(model, dataloaders)
