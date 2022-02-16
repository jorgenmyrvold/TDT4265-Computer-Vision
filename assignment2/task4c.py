import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer

# Function for simple training of different configurations of the network
def train_network(neurons_per_layer,
                  use_improved_sigmoid,
                  use_improved_weight_init,
                  use_momentum,
                  shuffle_data=True,
                  num_epochs=50,
                  momentum_gamma=.9,
                  batch_size=32):
    if use_momentum:
        learning_rate = .02
    else:
        learning_rate = .1

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    return train_history, val_history

if __name__ == "__main__":
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # 1 layers 64 neurons
    # train_history_1l_64n, val_history_1l_64n = train_network(
    #     neurons_per_layer=[64, 10],
    #     use_improved_sigmoid=False,
    #     use_improved_weight_init=False,
    #     use_momentum=False)

    # 1 layers 128 neurons
    # train_history_1l_128n, val_history_1l_128n = train_network(
    #     neurons_per_layer=[128, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True)
    #
    # # 1 layers 32 neurons
    # train_history_1l_32n, val_history_1l_32n = train_network(
    #     neurons_per_layer=[32, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True)
    #
    # # 1 layers 64 neurons - Only improved weights
    # train_history_weights, val_history_weights = train_network(
    #         neurons_per_layer=[64, 10],
    #         use_improved_sigmoid=False,
    #         use_improved_weight_init=True,
    #         use_momentum=False)
    #
    # # 1 layers 64 neurons - Only improved sigmoid
    # train_history_sigmoid, val_history_sigmoid = train_network(
    #     neurons_per_layer=[64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=False,
    #     use_momentum=False)
    #
    # # 1 layers 64 neurons - Only momentum
    # train_history_momentum, val_history_momentum = train_network(
    #     neurons_per_layer=[64, 10],
    #     use_improved_sigmoid=False,
    #     use_improved_weight_init=False,
    #     use_momentum=True)

    # 1 layers 64 neurons - All tricks
    # train_history_all_tricks, val_history_all_tricks = train_network(
    #     neurons_per_layer=[64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True)
    #
    # # 2 hidden layer 64 neurons per layer
    # train_history_2l_64n, val_history_2l_64n = train_network(
    #     neurons_per_layer=[64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True)
    #
    # # 3 hidden layer 64 neurons per layer
    # train_history_3l_64n, val_history_3l_64n = train_network(
    #     neurons_per_layer=[64, 64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True)
    #
    # # 4 hidden layer 64 neurons per layer
    # train_history_4l_64n, val_history_4l_64n = train_network(
    #     neurons_per_layer=[64, 64, 64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True)
    #
    # train_history_5l_64n, val_history_5l_64n = train_network(
    #     neurons_per_layer=[64, 64, 64, 64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True,
    #     num_epochs=5)
    #
    # train_history_6l_64n, val_history_6l_64n = train_network(
    #     neurons_per_layer=[64, 64, 64, 64, 64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True,
    #     num_epochs=5)
    #
    # train_history_7l_64n, val_history_7l_64n = train_network(
    #     neurons_per_layer=[64, 64, 64, 64, 64, 64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True,
    #     num_epochs=5)
    #
    # train_history_8l_64n, val_history_8l_64n = train_network(
    #     neurons_per_layer=[64, 64, 64, 64, 64, 64, 64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True,
    #     num_epochs=5)

    train_history_9l_64n, val_history_9l_64n = train_network(
        neurons_per_layer=[64, 64, 64, 64, 64, 64, 64, 64, 64, 10],
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=True,
        num_epochs=50)

    # 10 hidden layer 64 neurons per layer
    # train_history_10l_64n, val_history_10l_64n = train_network(
    #     neurons_per_layer=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10],
    #     use_improved_sigmoid=True,
    #     use_improved_weight_init=True,
    #     use_momentum=True)


    # Accuracy and loss plot in one
    # plt.subplot(1, 2, 1)
    # utils.plot_loss(train_history_1l_64n["loss"], "2 layer", npoints_to_average=10)
    # utils.plot_loss(train_history_2l_64n["loss"], "3 layer", npoints_to_average=10)
    # utils.plot_loss(train_history_10l_64n["loss"], "10 layer", npoints_to_average=10)
    # utils.plot_loss(train_history_sigmoid["loss"], "Improved sigmoid", npoints_to_average=10)
    # utils.plot_loss(train_history_weights["loss"], "Improved weights", npoints_to_average=10)
    # utils.plot_loss(train_history_momentum["loss"], "Momentum", npoints_to_average=10)
    # utils.plot_loss(train_history_all_tricks["loss"], "Momentum", npoints_to_average=10)
    # plt.ylim([0, .5])
    # plt.subplot(1, 2, 2)
    # plt.ylim([0.85, .99])
    # utils.plot_loss(val_history_1l_64n["accuracy"], "Baseline 2-layer")
    # utils.plot_loss(val_history_2l_64n["accuracy"], "3 Layer")
    # utils.plot_loss(val_history_10l_64n["accuracy"], "10 Layer")
    # utils.plot_loss(val_history_sigmoid["accuracy"], "Improved sigmoid")
    # utils.plot_loss(val_history_weights["accuracy"], "Improved weights")
    # utils.plot_loss(val_history_momentum["accuracy"], "Momentum")
    # utils.plot_loss(val_history_all_tricks["accuracy"], "All tricks")
    # plt.ylabel("Validation Accuracy")
    # plt.legend()
    # plt.show()

    # Accuracy plot tricks
    # plt.subplot()
    # plt.ylim([0.85, 1])
    # utils.plot_loss(val_history_1l_64n["accuracy"], "Baseline 2-layer")
    # utils.plot_loss(val_history_sigmoid["accuracy"], "Improved sigmoid")
    # utils.plot_loss(val_history_weights["accuracy"], "Improved weights")
    # utils.plot_loss(val_history_momentum["accuracy"], "Momentum")
    # utils.plot_loss(val_history_all_tricks["accuracy"], "All tricks")
    # plt.ylabel("Validation Accuracy")
    # plt.legend()
    # plt.show()

    # Accuracy plot multiple neurons
    # plt.subplot()
    # plt.ylim([0.85, 1])
    # utils.plot_loss(val_history_all_tricks["accuracy"], "64 Neuron")
    # utils.plot_loss(val_history_1l_32n["accuracy"], "32 Neuron")
    # utils.plot_loss(val_history_1l_128n["accuracy"], "128 Neuron")
    # plt.ylabel("Validation Accuracy")
    # plt.legend()
    # plt.show()

    # Accuracy plot multiple layers
    plt.subplot()
    plt.ylim([0, 1])
    # utils.plot_loss(val_history_all_tricks["accuracy"], "2 Layer")
    # utils.plot_loss(val_history_2l_64n["accuracy"], "3 Layer")
    # utils.plot_loss(val_history_3l_64n["accuracy"], "4 Layer")
    # utils.plot_loss(val_history_4l_64n["accuracy"], "5 Layer")
    # utils.plot_loss(val_history_5l_64n["accuracy"], "6 Layer")
    # utils.plot_loss(val_history_6l_64n["accuracy"], "7 Layer")
    # utils.plot_loss(val_history_7l_64n["accuracy"], "8 Layer")
    # utils.plot_loss(val_history_8l_64n["accuracy"], "9 Layer")
    utils.plot_loss(val_history_9l_64n["accuracy"], "10 Layer")
    # utils.plot_loss(val_history_10l_64n["accuracy"], "11 Layer")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
