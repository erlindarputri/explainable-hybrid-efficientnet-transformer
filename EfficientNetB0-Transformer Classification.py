# PREPARATION
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Subset
import pandas as pd
import time
from collections import Counter, defaultdict
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    zero_one_loss,
    roc_auc_score,
    roc_curve
)
from tqdm import tqdm
from PIL import Image
import random
from tabulate import tabulate
from sklearn.model_selection import StratifiedKFold, train_test_split
import copy
from sklearn.preprocessing import label_binarize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOADING THE DATASET
train_data_path = 'D:/2025 PENELITIAN/DATASET/Training'
test_data_path  = 'D:/2025 PENELITIAN/DATASET/Testing'

train_dataset_raw = datasets.ImageFolder(root=train_data_path)
test_dataset_raw  = datasets.ImageFolder(root=test_data_path)
class_names = train_dataset_raw.classes
num_classes = len(class_names)

if train_dataset_raw.class_to_idx != test_dataset_raw.class_to_idx:
    raise ValueError(
        "The mapping between the Training and Testing classes is different."
        "Make sure the folder names and class order are the same."
    )

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# EDA
def plot_class_distribution(dataset_raw):
    counts = Counter(label for _, label in dataset_raw)
    plt.figure(figsize=(8,4))
    sns.barplot(x=[class_names[i] for i in counts.keys()],
                y=[counts[i] for i in counts.keys()])
    plt.title("Class Distribution")
    plt.xticks(rotation=45)
    plt.show()

def image_size_stats(dataset_raw):
    widths, heights = [], []
    for path, _ in dataset_raw.imgs:
        w, h = Image.open(path).size
        widths.append(w); heights.append(h)
    print(f"Width:  min={np.min(widths)}, max={np.max(widths)}, mean={np.mean(widths):.1f}")
    print(f"Height: min={np.min(heights)}, max={np.max(heights)}, mean={np.mean(heights):.1f}")
    print()
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.histplot(widths, bins=20)
    plt.title("Width Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    
    plt.subplot(1,2,2)
    sns.histplot(heights, bins=20)
    plt.title("Height Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    
    plt.tight_layout(pad=3)
    plt.show()

def visualize_random_samples(dataset_raw, num_samples=8):
    indices = list(range(len(dataset_raw)))
    random.shuffle(indices)
    fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
    for i, idx in enumerate(indices[:num_samples]):
        img, lbl = dataset_raw[idx]
        axes[i].imshow(img); axes[i].set_title(class_names[lbl]); axes[i].axis('off')
    plt.show()

def visualize_one_per_class(dataset_raw):
    class_indices = defaultdict(list)
    for idx, (_, lbl) in enumerate(dataset_raw):
        class_indices[lbl].append(idx)
    fig, axes = plt.subplots(1, len(class_names), figsize=(3*len(class_names), 3))
    for cls, ax in zip(class_indices, axes):
        idx = random.choice(class_indices[cls])
        img, lbl = dataset_raw[idx]
        ax.imshow(img); ax.set_title(class_names[lbl]); ax.axis('off')
    plt.show()

print("=== Class Distribution ==="); plot_class_distribution(train_dataset_raw)
print()
print("=== Image Size Stats ==="); image_size_stats(train_dataset_raw)
print("=== Random Samples ==="); visualize_random_samples(train_dataset_raw)
print()
print("=== One per Class ==="); visualize_one_per_class(train_dataset_raw)

# PREPROCESSING
def compute_dataset_stats(dataset):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for imgs, _ in loader:
        imgs = imgs.to(torch.float64)

        channel_sum += imgs.sum(dim=(0, 2, 3))
        channel_sum_sq += (imgs ** 2).sum(dim=(0, 2, 3))
        total_pixels += imgs.size(0) * imgs.size(2) * imgs.size(3)

    mean = channel_sum / total_pixels
    variance = channel_sum_sq / total_pixels - mean ** 2
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    return mean.numpy(), std.numpy()


temp_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

temp_dataset = datasets.ImageFolder(
    root=train_data_path,
    transform=temp_transform
)

train_indices, val_indices = train_test_split(
    np.arange(len(temp_dataset)),
    test_size=0.25,
    stratify=np.array(temp_dataset.targets),
    random_state=RANDOM_SEED
)

mean, std = compute_dataset_stats(Subset(temp_dataset, train_indices))

print("\n=== Auto Dataset Normalization ===")
print("Mean :", mean)
print("Std  :", std)
print()

common_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist()),
])

train_transform = common_transform
test_transform  = common_transform

# DATASET DIVISION
full_train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
test_dataset  = datasets.ImageFolder(root=test_data_path,  transform=test_transform)

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

total = len(train_dataset) + len(val_dataset) + len(test_dataset)


print("\n=== Dataset Split Information ===")

print(f"Training   : {len(train_dataset)} ({len(train_dataset)/total*100:.2f}%)")
print(f"Validation : {len(val_dataset)} ({len(val_dataset)/total*100:.2f}%)")
print(f"Testing    : {len(test_dataset)} ({len(test_dataset)/total*100:.2f}%)")
print(f"Total Data : {total}")
print()

# DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# TRAINING MODEL
def train_model(model, train_loader, val_loader,
                criterion, optimizer,
                max_epochs=100, min_epochs=25, patience=5,
                verbose=True):

    training_start = time.time()
    model.to(device)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(max_epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{max_epochs}"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_loader(
            model, val_loader, criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {100*train_acc:.2f}% || "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {100*val_acc:.2f}%"
            )

        if epoch + 1 >= min_epochs:
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            elif epoch + 1 >= min_epochs:
                patience_counter += 1
            
            if epoch + 1 >= min_epochs and patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch  {epoch+1}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                break

    model.load_state_dict(best_model_weights)

    total_time = (time.time() - training_start) / 60

    if verbose:
        print(f"Total training time: {total_time:.2f} minutes")

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs
    }

    return model, history
    
def plot_training_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10,4))
    plt.plot(epochs, history["train_loss"], label="Training")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(epochs, history["train_acc"], label="Training")
    plt.plot(epochs, history["val_acc"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    

def evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss = total_loss / total
    acc  = correct / total

    return loss, acc    

# MODEL EVALUATION
def evaluate_with_confusion_table(model, loader, set_name):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in loader:

            imgs = imgs.to(device)

            outputs = model(imgs)
            
            probs = torch.softmax(outputs, dim=1).cpu().numpy()   
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n===== CONFUSION MATRIX ({set_name}) =====")

    plt.figure(figsize=(7,6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    
    plt.title(f"Confusion Matrix ({set_name})", fontsize=12)
    
    plt.xlabel("Predicted Label", fontsize=11, labelpad=15)
    plt.ylabel("True Label", fontsize=11, labelpad=15)

    plt.xticks(rotation=30)
    plt.yticks(rotation=0)
    
    plt.tight_layout(pad=2)
    
    plt.show()

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names]
    )

    print(tabulate(cm_df, headers="keys", tablefmt="grid"))

    print("\n===== TP FP TN FN PER CLASS =====")

    results = []

    for i, cls in enumerate(class_names):

        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        results.append([cls, TP, FP, TN, FN])

    table = pd.DataFrame(results, columns=["Class","TP","FP","TN","FN"])

    print(tabulate(table, headers="keys", tablefmt="grid", showindex=False))
    
    print("\n===== PRECISION, RECALL, F1-SCORE PER CLASS =====")
    
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class    = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class        = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    prf_results = []
    
    for i, cls in enumerate(class_names):
        prf_results.append([
            cls,
            f"{precision_per_class[i]:.4f}",
            f"{recall_per_class[i]:.4f}",
            f"{f1_per_class[i]:.4f}"
        ])
    
    prf_table = pd.DataFrame(
        prf_results,
        columns=["Class", "Precision", "Recall", "F1-Score"]
    )
    
    print(tabulate(prf_table, headers="keys", tablefmt="grid", showindex=False))
    
    print("\n===== AUC (ROC) =====")
    
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    
        auc_table = [
            ["AUC Macro",    f"{auc_macro:.4f}"],
            ["AUC Weighted", f"{auc_weighted:.4f}"]
        ]
    
        print(tabulate(auc_table, headers=["Metric","Value"], tablefmt="grid"))
    
    except Exception as e:
        print("AUC cannot be calculated:", e)
    
    print("\n===== ROC CURVE =====")
    
    y_true_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
    y_prob = np.array(all_probs)
    
    plt.figure(figsize=(7,6))
    
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_score = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
    
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc_score:.3f})")
    
    plt.plot([0,1], [0,1], linestyle='--')
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({set_name})")
    plt.legend()
    plt.grid()
    
    plt.show()
    
    acc  = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    error = zero_one_loss(all_labels, all_preds)

    print("\n===== PERFORMANCE METRICS =====")

    metrics_table = [
        ["Accuracy",  f"{acc:.2f}%"],
        ["Precision", f"{prec:.4f}"],
        ["Recall",    f"{rec:.4f}"],
        ["F1 Score",  f"{f1:.4f}"],
        ["Error",     f"{error:.4f}"]
    ]

    print(tabulate(metrics_table, headers=["Metric","Value"], tablefmt="grid"))   


# CNN-TRANSFORMER ARCHITECTURE
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel//reduction)
        self.fc2 = nn.Linear(channel//reduction, channel)
    def forward(self, x):
        b, c, _, _ = x.size()
        s = x.view(b, c, -1).mean(dim=2)
        e = F.relu(self.fc1(s))
        e = torch.sigmoid(self.fc2(e)).view(b, c, 1, 1)
        return x * e

class CNN_Transformer_Hybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        self.cnn = backbone.features

        self.se = SEBlock(1280)

        self.token = nn.Linear(1280, 512)

        self.drop1 = nn.Dropout(0.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            batch_first=True,
            dropout=0.1
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.drop2 = nn.Dropout(0.3)

        self.fc = nn.Linear(512, num_classes)

    
    def forward(self, x, return_features=False):

        f = self.cnn(x)
        f = self.se(f)
    
        if return_features:
            return f
    
        B, C, H, W = f.shape
    
        f = f.view(B, C, H * W).permute(0, 2, 1)
    
        t = self.drop1(self.token(f))
    
        t = self.transformer(t)
    
        t = t.mean(dim=1)
    
        out = self.fc(self.drop2(t))
    
        return out


counts        = Counter([lbl for _, lbl in train_dataset])
weights = torch.tensor([len(train_dataset)/(num_classes*counts[i])
                        for i in range(num_classes)], device=device)

model_cnn = CNN_Transformer_Hybrid(num_classes).to(device)

criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.1
)

optimizer = optim.Adam(
    model_cnn.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)


model_cnn, history_cnn = train_model(
    model_cnn,
    train_loader,
    val_loader,
    criterion,
    optimizer
)

plot_training_history(history_cnn)

evaluate_with_confusion_table(model_cnn, train_loader, "TRAINING")
evaluate_with_confusion_table(model_cnn, val_loader, "VALIDATION")
evaluate_with_confusion_table(model_cnn, test_loader, "TESTING")


# K-FOLD CROSS VALIDATION
print("\n===== K-FOLD CROSS VALIDATION =====")

k = 5

skf = StratifiedKFold(
    n_splits=k,
    shuffle=True,
    random_state=RANDOM_SEED
)

targets = np.array(train_dataset_raw.targets)

fold_acc = []
fold_loss = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(np.zeros(len(targets)), targets)
):

    print(f"\n========== FOLD {fold+1}/{k} ==========")

    fold_mean, fold_std = compute_dataset_stats(
        torch.utils.data.Subset(temp_dataset, train_idx)
    )

    fold_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(fold_mean.tolist(), fold_std.tolist()),
    ])

    fold_dataset = datasets.ImageFolder(
        root=train_data_path,
        transform=fold_transform
    )

    train_subset = torch.utils.data.Subset(
        fold_dataset, train_idx
    )

    val_subset = torch.utils.data.Subset(
        fold_dataset, val_idx
    )

    train_loader_k = torch.utils.data.DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True
    )

    val_loader_k = torch.utils.data.DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False
    )

    labels_fold = targets[train_idx]

    counts_fold = np.bincount(
        labels_fold,
        minlength=num_classes
    )

    if np.any(counts_fold == 0):
        raise ValueError(
            f"There are classes not found in the training fold {fold+1}"
        )

    weights_fold = torch.tensor(
        len(labels_fold) / (num_classes * counts_fold),
        dtype=torch.float32,
        device=device
    )

    criterion_k = nn.CrossEntropyLoss(
        weight=weights_fold,
        label_smoothing=0.1
    )

    model_k = CNN_Transformer_Hybrid(
        num_classes
    ).to(device)

    optimizer_k = optim.Adam(
        model_k.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )

    model_k, history_k = train_model(
        model_k,
        train_loader_k,
        val_loader_k,
        criterion_k,
        optimizer_k,
        max_epochs=100,
        min_epochs=25,
        patience=5,
        verbose=False
    )

    loss_k, acc_k = evaluate_loader(
        model_k,
        val_loader_k,
        criterion_k
    )

    print(f"Fold {fold+1} Validation Loss: {loss_k:.4f}")
    print(f"Fold {fold+1} Validation Accuracy: {acc_k*100:.2f}%")

    fold_loss.append(loss_k)
    fold_acc.append(acc_k)

print("\n===== 5-FOLD CROSS VALIDATION SUMMARY =====")

print(f"Mean Validation Accuracy : {np.mean(fold_acc)*100:.2f}%")
print(f"Std Validation Accuracy  : {np.std(fold_acc, ddof=1)*100:.2f}%")

print(f"Mean Validation Loss     : {np.mean(fold_loss):.4f}")
print(f"Std Validation Loss      : {np.std(fold_loss, ddof=1):.4f}")

plt.figure(figsize=(6,4))

fold_numbers = list(range(1, k+1))

plt.bar(fold_numbers, [acc*100 for acc in fold_acc])

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("K-Fold Cross Validation Accuracy")
plt.xticks(fold_numbers)

plt.grid(axis='y')

plt.show()



# GRAD-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.grad = None
        self.act = None

        self.h1 = self.layer.register_forward_hook(self.forward_hook)
        self.h2 = self.layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.act = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0].detach()

    def remove_hooks(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()

        out = self.model(x)

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1

        self.model.zero_grad()
        out.backward(gradient=one_hot)

        weights = self.grad.mean(dim=(2,3), keepdim=True)

        cam = (weights * self.act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx
 
    
# GRAD-CAM++
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.grad = None
        self.act = None

        self.h1 = self.layer.register_forward_hook(self.forward_hook)
        self.h2 = self.layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.act = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0].detach()

    def remove_hooks(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_idx=None):
        self.model.eval()

        out = self.model(x)

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1

        self.model.zero_grad()
        out.backward(gradient=one_hot)

        grad = self.grad
        act  = self.act

        grad2 = grad ** 2
        grad3 = grad ** 3

        eps = 1e-8

        sum_activation = act.sum(dim=(2,3), keepdim=True)

        alpha = grad2 / (2 * grad2 + sum_activation * grad3 + eps)

        weights = (alpha * F.relu(grad)).sum(dim=(2,3), keepdim=True)

        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224,224), mode='bilinear', align_corners=False)

        cam = F.avg_pool2d(cam, kernel_size=5, stride=1, padding=2)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx


def get_random_samples(dataset, target_class, top_k=10):
    class_indices = [
        idx for idx, (_, label) in enumerate(dataset)
        if label == target_class
    ]

    selected_indices = random.sample(
        class_indices,
        min(top_k, len(class_indices))
    )

    samples = [
        (0, dataset[idx][0], dataset[idx][1])
        for idx in selected_indices
    ]

    return samples


def visualize_results(model, samples, mean, std, class_names):

    mean_t = torch.tensor(mean).view(3,1,1)
    std_t  = torch.tensor(std).view(3,1,1)

    layer_gc   = model.cnn[-1]
    layer_gcpp = model.cnn[-1]

    for i, (score, img_tensor, label) in enumerate(samples):

        img_denorm = img_tensor * std_t + mean_t
        img_np = img_denorm.permute(1,2,0).numpy()
        img_np = np.clip(img_np, 0, 1)

        inp = img_tensor.unsqueeze(0).to(device)

        # GRAD-CAM
        gc = GradCAM(model, layer_gc)
        cam, _ = gc(inp, class_idx=label)
        gc.remove_hooks()

        cam = Image.fromarray(np.uint8(255 * cam))
        cam = cam.resize((img_np.shape[1], img_np.shape[0]))
        cam = np.array(cam)

        # GRAD-CAM++
        gcpp = GradCAMPlusPlus(model, layer_gcpp)
        cam_pp, _ = gcpp(inp, class_idx=label)
        gcpp.remove_hooks()

        cam_pp = Image.fromarray(np.uint8(255 * cam_pp))
        cam_pp = cam_pp.resize((img_np.shape[1], img_np.shape[0]))
        cam_pp = np.array(cam_pp)

        fig, ax = plt.subplots(1,3, figsize=(12,4))
        plt.subplots_adjust(wspace=0.08, hspace=0)

        ax[0].imshow(img_np)
        ax[0].set_title("Original", pad=8)
        ax[0].axis('off')
        
        ax[1].imshow(img_np)
        ax[1].imshow(cam, cmap='jet', alpha=0.4)
        ax[1].set_title("Grad-CAM", pad=8)
        ax[1].axis('off')
        
        ax[2].imshow(img_np)
        ax[2].imshow(cam_pp, cmap='jet', alpha=0.4)
        ax[2].set_title("Grad-CAM++", pad=8)
        ax[2].axis('off')

        plt.suptitle(f"{class_names[label]} | Image {i+1}", fontsize=14, y=1.00)
        
        plt.tight_layout()
        plt.show()
        

print("\n>>> Visualization of model attention using Grad-CAM and Grad-CAM++")


def normalize_class_name(name):
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")

no_tumor_aliases = {"notumor", "nontumor"}
no_tumor_indices = [
    i for i, name in enumerate(class_names)
    if normalize_class_name(name) in no_tumor_aliases
]

if len(no_tumor_indices) != 1:
    raise ValueError(
        f"The “No Tumor” class is not uniquely identified. Available classes: {class_names}. "
        "Set no_tumor_aliases to the name of the dataset folder."
    )

no_tumor_idx = no_tumor_indices[0]
tumor_classes = [i for i in range(num_classes) if i != no_tumor_idx]

for class_idx in tumor_classes:

    class_name = class_names[class_idx]

    print(f"\n===== CLASS: {class_name.upper()} =====")
    
    random_samples = get_random_samples(
        test_dataset,
        target_class=class_idx,
        top_k=10
    )
    
    visualize_results(
        model_cnn,
        random_samples,
        mean,
        std,
        class_names
    )
    

class_name = class_names[no_tumor_idx]
print(f"\n===== CLASS: {class_name.upper()} =====")

samples = []

for img, label in test_dataset:
    if label == no_tumor_idx:
        samples.append((0, img, label))

selected = random.sample(
    samples,
    min(10, len(samples))
)

visualize_results(
    model_cnn,
    selected,
    mean,
    std,
    class_names
)
