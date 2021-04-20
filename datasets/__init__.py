from .mnists import MNIST, FashionMNIST


def load_dataset(args):
    data_module = {
        'mnist':  MNIST,
        'fashion_mnist': FashionMNIST,
    }[args.dataset_name](args)

    img_size = {
        'mnist': [1, 28, 28],
        'fashion_mnist': [1, 28, 28],
    }[args.dataset_name]
    with args.unlocked():
        args.image_size = img_size
    return data_module, args
