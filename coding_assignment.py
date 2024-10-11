import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import torchvision
from torchvision import transforms
import lightning as L
import wandb

#import the model
clip_model_path = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_path)
clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

run = wandb.init(project="coding_assignment")

#freeze the model weights
for parameter in clip_model.parameters():
    parameter.requires_grad = False
    
clip_model.eval()
    
#define the transformation of the dataset
transform = transforms.Compose([transforms.Lambda(lambda img: img.convert("RGB")), transforms.Resize((224,224)), transforms.ToTensor()])

#load the dataset
dataset = torchvision.datasets.Caltech101(root="./data", download=True, target_type="category", transform=transform)

#split the dataset into train and test
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

class ClipClassifier(L.LightningModule):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.linear1 = nn.Linear(self.clip_model.config.projection_dim, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        with torch.no_grad():
            inputs = clip_processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
            embeddings = self.clip_model.get_image_features(**inputs)
            x = nn.functional.relu(self.linear1(embeddings))
            x = self.dropout(x)
            x = nn.functional.relu(self.linear2(x))
            x = self.dropout(x)
            return self.linear3(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = nn.functional.cross_entropy(logits, labels)
        run.log({'train_loss': loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
    
    def on_train_epoch_end(self):
        test_evaluation(self, test_loader)
        
def test_evaluation(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = total_loss / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    run.log({'test_loss':avg_loss})
    run.log({'test_accuracy': accuracy})
      
num_classes = 101
model = ClipClassifier(clip_model, num_classes)

trainer = L.Trainer(max_epochs = 10, logger=run, log_every_n_steps=10)
trainer.fit(model, train_loader, test_loader)

wandb.finish()