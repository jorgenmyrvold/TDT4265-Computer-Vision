import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from tabulate import tabulate

# Default model from task 2
# Kernel_size = 5
# (conv-relu-pool)x3 → (affine)x1 → softmax
# Initial model task 2a
class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (16, 16)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (8, 8)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # new dim (4, 4)
        )
        self.num_output_features = 4 * 4 * 128
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(-1, self.num_output_features)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


# (conv-relu-conv-relu-pool)x3 → (affine)x1 → softmax
# kernel_size = 5
# Worst 75% model
class Model_1(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (16, 16)
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (8, 8)
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # new dim (4, 4)
        )
        self.num_output_features = 4 * 4 * 128
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(-1, self.num_output_features)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


# Kernel_size = 5
# (conv-relu-pool)x3 → (affine)x1 → softmax
# Batch normalization after each convolution
# Best 75% accuracy model
class Model_2(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (16, 16)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (8, 8)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # new dim (4, 4)
        )

        self.num_output_features = 4 * 4 * 128
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(-1, self.num_output_features)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

# Kernel_size = 5
# (conv-relu-pool)x3 → (affine)x1 → softmax
# Batch normalization after each convolution
# 80% test accuracy
class Model_3(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (16, 16)
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (8, 8)
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # new dim (4, 4)
        )
        self.num_output_features = 4 * 4 * 128
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(-1, self.num_output_features)
        out = self.classifier(out)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def get_final_result(model, dataloaders, print_table=True, loss_criterion=nn.CrossEntropyLoss()):
    dataloaders_name = ["Training", "Validation", "Test"]
    results = []

    for i, loader in enumerate(dataloaders):
        loss, acc = compute_loss_and_accuracy(loader, model, loss_criterion)
        row = [dataloaders_name[i], loss, acc]
        results.append(row)

    if print_table:
        print('\n' + tabulate(results, headers=['Dataset', 'Loss', 'Accuracy'], tablefmt='orgtbl', floatfmt=".4f") + '\n')
        
    return results

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def plot_validation_accuracy(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(8, 5))
    plt.title("Accuracy")
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Accuracy')
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def plot_training_validation_loss(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(8, 5))
    plt.title("Cross Entropy Loss")
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Cross Entropy Loss')
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def create_comparison_plots(trainers: list, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    
    for i, trainer in enumerate(trainers):
        utils.plot_loss(trainer.train_history["loss"], label=f"Training loss - Model {i+1}", npoints_to_average=10)
        utils.plot_loss(trainer.validation_history["loss"], label=f"Validation loss - Model {i+1}")

    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")

    for trainer in trainers:
        utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")

    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def create_comparison_loss_plots(trainers: list, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(8, 5))
    plt.title("Cross Entropy Loss")
    
    for i, trainer in enumerate(trainers):
        utils.plot_loss(trainer.train_history["loss"], label=f"Training loss - Model {i+1}", npoints_to_average=10)
        utils.plot_loss(trainer.validation_history["loss"], label=f"Validation loss - Model {i+1}")

    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc)
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(2)
    epochs = 10
    batch_size = 64
    early_stop_count = 4

    # select_model chooses which model to use and train.
    # 0 - base model from task 2
    # 1 - First model with 75% accuracy
    # 2 - Second model with 75% accuracy
    # 3 - 80% accuracy model
    select_model = 3

    if select_model == 0:
        # Task 2 - Model based on table 1 from assignment
        dataloaders = load_cifar10(batch_size, data_augmentation=False)
        model = ExampleModel(image_channels=3, num_classes=10)
        trainer = Trainer(
            batch_size=batch_size,
            learning_rate=5e-2,
            early_stop_count=early_stop_count,
            epochs=epochs,
            model=model,
            dataloaders=dataloaders,
            optimizer='SGD'
        )
    
    elif select_model == 1:
        # Task 3a - First model with 75% accuracy
        dataloaders = load_cifar10(batch_size)
        model = Model_1(image_channels=3, num_classes=10)
        trainer = Trainer(
            batch_size=batch_size,
            learning_rate=5e-2,
            early_stop_count=early_stop_count,
            epochs=epochs,
            model=model,
            dataloaders=dataloaders,
            optimizer='SGD'
        )

    elif select_model == 2:
        # Task 3a - Second model with 75% accuracy
        dataloaders = load_cifar10(batch_size)
        model = Model_2(image_channels=3, num_classes=10)
        trainer = Trainer(
            batch_size=batch_size,
            learning_rate=5e-2,
            early_stop_count=early_stop_count,
            epochs=epochs,
            model=model,
            dataloaders=dataloaders,
            optimizer='SGD'
        )
    
    elif select_model == 3:
        # Task 3e - Model with 80% accuracy
        dataloaders = load_cifar10(batch_size)
        model = Model_3(image_channels=3, num_classes=10)
        trainer = Trainer(
            batch_size=batch_size,
            learning_rate=1e-1,
            early_stop_count=early_stop_count,
            epochs=epochs,
            model=model,
            dataloaders=dataloaders,
            optimizer='SGD'
        )

    trainer.train()

    plot_validation_accuracy(trainer, 'val_acc')
    plot_training_validation_loss(trainer, 'train_val_loss')

    get_final_result(model, dataloaders)

if __name__ == "__main__":
    main()
