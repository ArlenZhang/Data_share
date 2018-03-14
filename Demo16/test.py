import utils
mnist_folder = '../../data/mnist'
train_data, test_data = utils.get_mnist_dataset(1, mnist_folder=mnist_folder)
print(train_data)
