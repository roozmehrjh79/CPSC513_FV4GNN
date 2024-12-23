# ----- Imports ----- #
import time
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import KarateClub
from gnn_models import GCN


# ----- User-defined parameters ----- #
DEBUG = False
FORCE_USE_CPU = True
SAVE_MODEL = False
MODEL_NAME = "gcn_karate.pth"

# ----- Checking if GPU is available ----- #
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

if FORCE_USE_CPU:
    device = torch.device('cpu')
    
# ----- Dataset ----- #
dataset = KarateClub()
data = dataset[0].to(device)

# Print dataset info
print(dataset)
print("------------")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")
if DEBUG:
    print(f"dataset.x: {dataset.x}")
    print(f"Mean of label for training set: {torch.mean(data.y.float()[data.train_mask])}")

# Set custom test mask
test_mask = torch.ones(data.x.shape[0], dtype=torch.bool)

# ----- Setting up the model and train & test routines ----- #
model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=8,
    out_channels=dataset.num_classes,
).to(device)

optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # only perform weight-decay on first convolution.

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)
    
@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

# ----- Training & testing ----- #
best_val_acc = test_acc = 0
times = []
for epoch in range(1, 200 + 1):
    start = time.time()
    loss = train()
    train_acc, tmp_test_acc = test()
    test_acc = tmp_test_acc
    if epoch % 10 == 0:
        print(f"Epoch {epoch:>3} | Loss: {loss:.2f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

# Saving the model
if SAVE_MODEL:
    torch.save(model.state_dict(), f"./models/torch/{MODEL_NAME}")
