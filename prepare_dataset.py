from pathlib import Path
import shutil
import random


PROJECT_DIR = Path(__file__).parent

SOURCE_DIR = (
    PROJECT_DIR
    / "PlantVillage-Dataset"
    / "raw"
    / "color"
)

DATASET_DIR = (
    PROJECT_DIR
    / "dataset"
)

TRAIN_DIR = (
    DATASET_DIR
    / "train"
)

VALIDATION_DIR = (
    DATASET_DIR
    / "validation"
)


TRAIN_RATIO = 0.80

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def prepare_dataset():

    if not SOURCE_DIR.exists():

        print(
            "ERROR: PlantVillage dataset not found."
        )

        print(
            f"Expected location: {SOURCE_DIR}"
        )

        return

    print(
        "\nPlantVillage dataset found."
    )

    print(
        f"Source: {SOURCE_DIR}"
    )

    print(
        "\nFinding crop disease classes...\n"
    )

    class_directories = sorted(
        [
            folder
            for folder in SOURCE_DIR.iterdir()
            if folder.is_dir()
        ]
    )

    if not class_directories:

        print(
            "ERROR: No class folders found."
        )

        return

    print(
        f"Found {len(class_directories)} classes."
    )

    for folder in class_directories:

        print(
            f"  - {folder.name}"
        )

    print(
        "\nPreparing train and validation folders..."
    )

    if DATASET_DIR.exists():

        print(
            "\nRemoving old dataset..."
        )

        shutil.rmtree(
            DATASET_DIR
        )

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_train = 0

    total_validation = 0

    class_names = []

    for class_directory in class_directories:

        class_name = (
            class_directory.name
        )

        class_names.append(
            class_name
        )

        image_files = []

        for file in class_directory.iterdir():

            if file.is_file():

                if file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:

                    image_files.append(
                        file
                    )

        random.shuffle(
            image_files
        )

        total_images = len(
            image_files
        )

        if total_images == 0:

            print(
                f"\nWARNING: No images found in {class_name}"
            )

            continue

        train_count = int(
            total_images * TRAIN_RATIO
        )

        train_images = (
            image_files[:train_count]
        )

        validation_images = (
            image_files[train_count:]
        )

        train_class_dir = (
            TRAIN_DIR
            / class_name
        )

        validation_class_dir = (
            VALIDATION_DIR
            / class_name
        )

        train_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        validation_class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for image_file in train_images:

            destination = (
                train_class_dir
                / image_file.name
            )

            shutil.copy2(
                image_file,
                destination
            )

        for image_file in validation_images:

            destination = (
                validation_class_dir
                / image_file.name
            )

            shutil.copy2(
                image_file,
                destination
            )

        total_train += len(
            train_images
        )

        total_validation += len(
            validation_images
        )

        print(
            f"{class_name}: "
            f"{len(train_images)} train, "
            f"{len(validation_images)} validation"
        )

    class_names_file = (
        PROJECT_DIR
        / "model"
        / "class_names.txt"
    )

    class_names_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        class_names_file,
        "w",
        encoding="utf-8"
    ) as file:

        for class_name in class_names:

            file.write(
                class_name + "\n"
            )

    print(
        "\n========================================"
    )

    print(
        "DATASET PREPARATION COMPLETED"
    )

    print(
        "========================================"
    )

    print(
        f"Number of classes: {len(class_names)}"
    )

    print(
        f"Training images: {total_train}"
    )

    print(
        f"Validation images: {total_validation}"
    )

    print(
        f"Total images: "
        f"{total_train + total_validation}"
    )

    print(
        f"\nClass names saved to:"
    )

    print(
        class_names_file
    )


if __name__ == "__main__":

    prepare_dataset()