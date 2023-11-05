from torchvision.models import (efficientnet_v2_m, EfficientNet_V2_M_Weights,
                                efficientnet_v2_s, EfficientNet_V2_S_Weights,
                                efficientnet_v2_l, EfficientNet_V2_L_Weights,
                                vit_b_16, ViT_B_16_Weights,
                                mobilenet_v3_small, MobileNet_V3_Small_Weights,
                                mobilenet_v3_large, MobileNet_V3_Large_Weights
                                )

dict_models = {
    "efficient_net_small": {
        "name": efficientnet_v2_s,
        "weights": EfficientNet_V2_S_Weights,
        "last_layer_name": "classifier",
        "last_layer_dim": 1280
    },
    "efficient_net_medium": {
        "name": efficientnet_v2_m,
        "weights": EfficientNet_V2_M_Weights,
        "last_layer_name": "classifier",
        "last_layer_dim": 1280
    },
    "efficient_net_large": {
        "name": efficientnet_v2_l,
        "weights": EfficientNet_V2_L_Weights,
        "last_layer_name": "classifier",
        "last_layer_dim": 1280
    },
    "vision_transformer": {
        "name": vit_b_16,
        "weights": ViT_B_16_Weights,
        "last_layer_name": "heads",
        "last_layer_dim": 768
    },
    "mobilenet_v3_small": {
        "name": mobilenet_v3_small,
        "weights": MobileNet_V3_Small_Weights,
        "last_layer_name": "classifier",
        "last_layer_dim": 576
    },
    "mobilenet_v3_large": {
        "name": mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights,
        "last_layer_name": "classifier",
        "last_layer_dim": 960
    }
}
