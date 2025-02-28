LAYER_NAMES_BY_MODEL = {
    'lenet_cmnist': [
        "input_identity", 
        "features.0",
        "features.1", 
        "features.2", 
        "features.3"
        ],
    'lenet_cifar10': [
        "input_identity", 
        "features.0",
        "features.1", 
        "features.2", 
        "features.3"
        ],
    'vgg11': [
        "input_identity", 
        "features.0",
        "features.3",
        "features.6",
        "features.8",
        "features.11",
        "features.13",
        "features.16",
        "features.18"
        ],
    'vgg16': [
        "input_identity", 
        "features.0",
        "features.2",
        "features.5",
        "features.7",
        "features.10",
        "features.12",
        "features.14",
        "features.17",
        "features.19",
        "features.21",
        "features.24",
        "features.26",
        "features.28"
        ],
    'vgg16_with_relu': [
        "features.0",
        "features.1",
        "features.2",
        "features.3",
        "features.5",
        "features.6",
        "features.7",
        "features.8",
        "features.10",
        "features.11",
        "features.12",
        "features.13",
        "features.14",
        "features.15",
        "features.17",
        "features.18",
        "features.19",
        "features.20",
        "features.21",
        "features.22",
        "features.24",
        "features.25",
        "features.26",
        "features.27",
        "features.28",
        "features.29"
        ],
    'resnet': [
        "input_identity", 
        "identity_0",
        "identity_1",
        "identity_2",
        "last_conv"
        ],
    'resnext': [
        "input_identity", 
        "identity_0",
        "identity_1",
        "identity_2",
        "last_conv"
        ],
    'rexnet': [
        "input_identity", 
        "identity_12",
        "identity_13",
        "identity_14",
        "identity_15",
        "last_conv"
        ],
    'efficientnet_b0': [
        "input_identity", 
        "identity_0",
        "identity_1",
        "identity_2",
        "identity_3",
        "identity_4",
        "identity_5",
        "identity_6",
        "identity_7",
        "last_conv"
        ],
        'efficientnet_v2': [
        "input_identity", 
        "identity_0",
        "identity_1",
        "identity_2",
        "identity_3",
        "identity_4",
        "identity_5",
        "identity_6",
        "last_conv"
        ],
        "vit": [
            # "identity_8",
            # "identity_9",
            # "identity_10",
            "inspection_layer"
        ],
        "swin_former": [
            # "identity_0",
            # "identity_1",
            # "identity_2",
            "inspection_layer"
        ],
        "metaformer": [
            # "identity_0",
            # "identity_1",
            # "identity_2",
            "inspection_layer"
        ]
}