import torchreid

# Build the model architecture and load pre-trained weights
model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=751, # Number of unique IDs in Market1501
    pretrained=True, # Will automatically download the ImageNet pre-trained backbone
    loss="softmax"
)