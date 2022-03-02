import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from tabulate import tabulate

# Suggested architectures
# (conv-relu-pool)xN → (affine)xM → softmax
# (conv-relu-conv-relu-pool)xN → (affine)xM → softmax
# (batchnorm-relu-conv)xN → (affine)xM → softmax

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

# Kernel_size = 3 
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
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (16, 16)
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # new dim (8, 8)
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
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
# (conv-relu-pool)x3 → (affine)x2 → softmax
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
            nn.Linear(64, 64),
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


def main():
    # Set the random generator seed (parameters, shuffling etc)
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)

    epochs = 10
    batch_size = 64
    early_stop_count = 4

    dataloaders = load_cifar10(batch_size)

    # Note on different optimizers. Relevant HP are listed below
    # SGD: lr
    # SGD_momentum: lr, momentum
    # Adam_regularization: lr, l2_regularization

    select_model = 3
    model = None

    if select_model == 0:
        # Default model from task 2
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
        model = Model_2(image_channels=3, num_classes=10)
        trainer = Trainer(
            batch_size=batch_size,
            learning_rate=5e-2,
            early_stop_count=early_stop_count,
            epochs=epochs,
            model=model,
            dataloaders=dataloaders,
            regularization=0.5,
            optimizer='Adam_regularization'
        )
    
    elif select_model == 3:
        model = Model_3(image_channels=3, num_classes=10)
        trainer = Trainer(
            batch_size=batch_size,
            learning_rate=5e-2,
            early_stop_count=early_stop_count,
            epochs=epochs,
            model=model,
            dataloaders=dataloaders,
            momentum=0.9,
            optimizer='SGD_momentum'
        )

    trainer.train()
    create_plots(trainer, "task2")

    get_final_result(model, dataloaders)

if __name__ == "__main__":
    main()
