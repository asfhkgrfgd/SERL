from datasets import load_dataset
from transformers import AutoImageProcessor
from torchvision.transforms import Compose, RandomResizedCrop, ColorJitter

def get_data_loaders(dataset_name,
                     processor_name="microsoft/swinv2-base-patch4-window16-256",
                     cache_dir="/date/wqq/swinv2/data_cache"):
    print(f"Loading dataset: {dataset_name} ")
    dataset = load_dataset(dataset_name,cache_dir=cache_dir)
    num_classes = dataset["train"].features["label"].num_classes
    print(f"Dataset loaded: {dataset_name}")

    if "validation" not in dataset and "train" in dataset:
        dataset_split = dataset["train"].train_test_split(test_size=0.1)
        dataset = {
            "train": dataset_split["train"],
            "validation": dataset_split["test"],
            "test": dataset.get("test", dataset_split["test"])
        }

    processor = AutoImageProcessor.from_pretrained(processor_name)


    size = processor.size.get("shortest_edge") or (processor.size["height"], processor.size["width"])
    tfms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5)]) 

    def transform_fn(batch):
        images = batch.get("image") or batch.get("img")
        if images is None:
            raise ValueError("No 'image'or'img'")
        images = [tfms(img.convert("RGB")) for img in images]
        encoded = processor(images, do_resize=True, return_tensors="pt")
        batch["pixel_values"] = encoded["pixel_values"]
        return batch
        
    dataset["train"].set_transform(transform_fn)
    dataset["validation"].set_transform(transform_fn)
    dataset["test"].set_transform(transform_fn)

    return dataset["train"], dataset["validation"], dataset["test"], num_classes

if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        dataset_name="fashion_mnist"
    )
    print(f"Number of classes: {num_classes}")
    print(train_loader)