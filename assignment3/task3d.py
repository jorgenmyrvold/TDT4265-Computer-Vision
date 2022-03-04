import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from tabulate import tabulate
from task2 import get_final_result

# Default model from task 2
# Kernel_size = 5
# (conv-relu-pool)x3 → (affine)x1 → softmax
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


# def create_comparison_loss_plots(baseline_trainer: Trainer, comp_trainer: Trainer, name: str):
def create_comparison_loss_plots(trainers: Trainer, name: str, include_training_loss=False):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(8, 5))
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Cross Entropy Loss')

    for i, trainer in enumerate(trainers):
        if include_training_loss:
            utils.plot_loss(trainer.train_history["loss"], label=f"Training loss - {i}", npoints_to_average=10)
        utils.plot_loss(trainer.validation_history["loss"], label=f"Validation loss - {i}")

    
    # utils.plot_loss(baseline_trainer.train_history["loss"], label="Training loss - Baseline", npoints_to_average=10)
    # utils.plot_loss(baseline_trainer.validation_history["loss"], label="Validation loss - Baseline")
    # utils.plot_loss(comp_trainer.train_history["loss"], label="Training loss - Data aug.", npoints_to_average=10)
    # utils.plot_loss(comp_trainer.validation_history["loss"], label="Validation loss - Data aug.")

    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc)
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, data_augmentation=False)
    dataloaders_1 = load_cifar10(batch_size, data_augmentation=True)

    # Task 3d - Create comparison plot of two different models
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

    model_1 = ExampleModel(image_channels=3, num_classes=10) # Data augmentation
    trainer_1 = Trainer(
        batch_size=batch_size,
        learning_rate=5e-2,
        early_stop_count=early_stop_count,
        epochs=epochs,
        model=model_1,
        dataloaders=dataloaders_1,
        optimizer='SGD'
    )

    model_2 = Model_2(image_channels=3, num_classes=10) # Batch normalization
    trainer_2 = Trainer(
        batch_size=batch_size,
        learning_rate=5e-2,
        early_stop_count=early_stop_count,
        epochs=epochs,
        model=model_2,
        dataloaders=dataloaders,
        optimizer='SGD'
    )

    model_3 = Model_1(image_channels=3, num_classes=10) # New architecthure
    trainer_3 = Trainer(
        batch_size=batch_size,
        learning_rate=5e-2,
        early_stop_count=early_stop_count,
        epochs=epochs,
        model=model_3,
        dataloaders=dataloaders,
        optimizer='SGD'
    )

    trainer.train()
    trainer_1.train()
    trainer_2.train()
    trainer_3.train()

    create_comparison_loss_plots(trainers=[trainer, trainer_1, trainer_2, trainer_3], name="3d_horizontal_flip_comparason_all")
    create_comparison_loss_plots(trainers=[trainer, trainer_3], name="3d_horizontal_flip_comparason", include_training_loss=True)

    print('Original model', end='')
    get_final_result(model, dataloaders)

    print('With data augmentation', end='')
    get_final_result(model_1, dataloaders_1)

    print('With new architecture', end='')
    get_final_result(model_2, dataloaders)

    print('With batch normalization', end='')
    get_final_result(model_3, dataloaders)

if __name__ == "__main__":
    main()
