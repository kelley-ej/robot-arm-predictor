import os
import numpy as np

SOURCE_DIR = "/work/cssema416/202710/13/src/bc_data"
OUTPUT_DIR = "demo_data"

NUM_SAMPLES = 50
SEED = 42

rng = np.random.default_rng(SEED)

splits = ["train", "val", "test"]

for split in splits:
    source_split = os.path.join(SOURCE_DIR, split)
    output_split = os.path.join(OUTPUT_DIR, split)

    os.makedirs(output_split, exist_ok=True)

    print(f"\nProcessing {split}...")

    top_images = np.load(
        os.path.join(source_split, "image.npy"),
        mmap_mode="r"
    )

    wrist_images = np.load(
        os.path.join(source_split, "image_wrist.npy"),
        mmap_mode="r"
    )

    states = np.load(
        os.path.join(source_split, "states.npy"),
        mmap_mode="r"
    )

    actions = np.load(
        os.path.join(source_split, "actions.npy"),
        mmap_mode="r"
    )

    num_available = len(states)
    num_samples = min(NUM_SAMPLES, num_available)

    indices = rng.choice(
        num_available,
        size=num_samples,
        replace=False
    )
    indices.sort()

    np.save(
        os.path.join(output_split, "image.npy"),
        np.array(top_images[indices])
    )

    np.save(
        os.path.join(output_split, "image_wrist.npy"),
        np.array(wrist_images[indices])
    )

    np.save(
        os.path.join(output_split, "states.npy"),
        np.array(states[indices])
    )

    np.save(
        os.path.join(output_split, "actions.npy"),
        np.array(actions[indices])
    )

    np.save(
        os.path.join(output_split, "original_indices.npy"),
        indices
    )

    print(f"Saved {num_samples} samples to {output_split}")


print("\nDemo dataset creation complete!")