import pickle
from lib.dataloader import prepare_data
from lib.model import MultiViewFusion
from lib.trainer import train_model
from lib.example import Example

dataset_name = "lego-classify-multi-01"
dataset_path = f"datasets/{dataset_name}.pkl"

print(f"Loading dataset {dataset_path}...")
with open(dataset_path, 'rb') as f:
    examples = pickle.load(f)
train_loader, val_loader, part_to_idx = prepare_data(examples, batch_size=32)
num_classes = len(part_to_idx)
print(f"{num_classes} classes, {len(examples)} examples")

model = MultiViewFusion(num_classes=num_classes)
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100
)
