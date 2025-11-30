import argparse
import torch
import os
import json
from torch import nn
from tqdm import tqdm
from load_data import get_data_loaders
from transformers import AutoModelForImageClassification
from RLoss import structure_entropy_reg
from datetime import datetime
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.dim() == 0 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    else:
        return obj

def evaluate(model, dataset, criterion, device,num_classes, batch_size=32, args=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = len(dataset) // batch_size

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            pixel_values = batch["pixel_values"].to(device)
            labels = torch.tensor(batch["label"]).to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            # Apply entropy loss if specified
            if args.lamda > 0:
                # probs = torch.softmax(outputs.logits, dim=1)
                entropy_loss = structure_entropy_reg(outputs.logits,labels,num_classes)
                total_loss += args.lamda * entropy_loss

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / num_batches
    accuracy = correct / total
    return avg_loss, accuracy


def train(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("--------------------------")
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset, num_classes = get_data_loaders(
        dataset_name=args.dataset,
        processor_name=args.model_name,
    )

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        cache_dir="/date/wqq/model_cache"
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    
    print(f"Model loaded: {args.model_name} with {num_classes} classes")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("output", exist_ok=True)
    log_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(train_dataset) // args.batch_size

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        for i in tqdm(range(0, len(train_dataset), args.batch_size), desc="Training", unit="batch"):
            batch = train_dataset[i : i + args.batch_size]
            pixel_values = batch["pixel_values"].to(device)
            labels = torch.tensor(batch["label"]).to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = criterion(outputs.logits, labels)
            print(f"Batch {i//args.batch_size + 1}/{num_batches}, Loss: {loss.item():.4f}")

            # Apply entropy loss if specified
            if args.lamda > 0:
                # probs = torch.softmax(outputs.logits, dim=1)
                entropy_loss = structure_entropy_reg(outputs.logits,labels,num_classes)
                print
                loss += args.lamda * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / num_batches
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_dataset, criterion, device,num_classes, batch_size=args.batch_size,args=args)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        # 保存日志信息
        log_data.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        log_filename = f"./output/ser/{args.model_name}_{args.dataset}_{timestamp}.json"
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        with open(log_filename, "w") as f:
            json.dump(convert_to_serializable(log_data), f, indent=4)

    torch.save(model.state_dict(), args.save_path)
    print(f"\nModel saved to: {args.save_path}")

    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_dataset, criterion, device,num_classes, batch_size=args.batch_size,args=args)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="swinv2 Fine-tune Image Classification Transformer")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset: mnist, cifar10, fashion_mnist")
    parser.add_argument("--model_name", type=str, default="microsoft/swinv2-base-patch4-window16-256,microsoft/resnet-50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--save_path", type=str, default="./model.pth")
    parser.add_argument("--lamda", type=float, default=0.01, help="Weight for entropy loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
