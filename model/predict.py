import torch
import torchvision
import numpy as np


CHECKPOINT = "/home/kelleyej/13/robot-arm-web/model/robot_finetuned.pth"

DEVICE = torch.device("cpu")


def load_feature_extractor(checkpoint, prefix):
    """
    Load one fine-tuned VGG feature branch.
    Only one branch is kept in memory at a time.
    """

    features = torchvision.models.vgg16(
        weights=None
    ).features

    branch_state = {}

    for key, value in checkpoint["model_state_dict"].items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            branch_state[new_key] = value

    features.load_state_dict(branch_state)
    features.eval()

    return features


def preprocess_image(image):
    """
    Convert an image into the format used during training.

    Input:
        image: numpy array [H, W, 3]

    Output:
        tensor [1, 3, 224, 224]
    """

    image = torch.from_numpy(
        np.asarray(image)
    )

    # H,W,C -> C,H,W
    image = image.permute(2, 0, 1).float() / 255.0

    # Add batch dimension
    image = image.unsqueeze(0)

    # Resize to VGG input size
    image = torch.nn.functional.interpolate(
        image,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    # Same normalization used during training
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(1, 3, 1, 1)

    image = (image - mean) / std

    return image


def extract_features(features, image):
    """
    Run one image through one VGG feature extractor.
    """

    with torch.inference_mode():

        x = features(image)

        # VGG16 with 224x224 input gives 512 x 7 x 7
        x = torch.flatten(x, 1)

    return x


def predict_robot_action(top_image, wrist_image, state):
    """
    Predict the 7 robot actuator commands.

    Parameters
    ----------
    top_image : numpy array
        Top camera image, shape [H,W,3]

    wrist_image : numpy array
        Wrist camera image, shape [H,W,3]

    state : array-like
        15-element robot state vector

    Returns
    -------
    prediction : numpy array
        7 predicted actuator commands
    """

    print("Loading model checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    print("Checkpoint loaded.")


    # =========================================================
    # PREPROCESS IMAGES
    # =========================================================

    top_image = preprocess_image(top_image)
    wrist_image = preprocess_image(wrist_image)

    state = torch.tensor(
        np.asarray(state),
        dtype=torch.float32
    ).reshape(1, -1)


    # =========================================================
    # TOP CAMERA
    # =========================================================

    print("Processing top camera...")

    features1 = load_feature_extractor(
        checkpoint,
        "features1."
    )

    top_features = extract_features(
        features1,
        top_image
    )

    print("Top features:", top_features.shape)

    # Release VGG branch 1
    del features1
    del top_image


    # =========================================================
    # WRIST CAMERA
    # =========================================================

    print("Processing wrist camera...")

    features2 = load_feature_extractor(
        checkpoint,
        "features2."
    )

    wrist_features = extract_features(
        features2,
        wrist_image
    )

    print("Wrist features:", wrist_features.shape)

    # Release VGG branch 2
    del features2
    del wrist_image


    # =========================================================
    # LOAD DECODER
    # =========================================================

    print("Loading decoder...")

    regressor = torch.nn.Sequential(
        torch.nn.Linear(
            2 * 25088 + checkpoint["state_dim"],
            128
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(
            128,
            checkpoint["action_dim"]
        )
    )

    regressor_state = {}

    for key, value in checkpoint["model_state_dict"].items():

        if key.startswith("regressor."):
            new_key = key[len("regressor."):]
            regressor_state[new_key] = value

    regressor.load_state_dict(regressor_state)
    regressor.eval()


    # =========================================================
    # PREDICTION
    # =========================================================

    print("Generating prediction...")

    with torch.inference_mode():

        combined = torch.cat(
            [
                top_features,
                wrist_features,
                state
            ],
            dim=1
        )

        prediction = regressor(combined)


    # Convert PyTorch tensor → NumPy
    prediction = prediction.squeeze(0).numpy()

    print("Prediction:", prediction)

    return prediction