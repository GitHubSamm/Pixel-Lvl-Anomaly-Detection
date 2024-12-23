import os
import argparse
import random
import string
import numpy as np
import cv2
from PIL import Image
from glob import glob


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, confusion_matrix


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def prep_dirs(root):
    """Prepare directories for saving samples and source code."""
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    return sample_path, source_code_save_path


def min_max_norm(image):
    """Normalize image to [0, 1] range."""
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min + 1e-8)


def cvt2heatmap(gray):
    """Convert grayscale to heatmap."""
    return cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)


def heatmap_on_image(heatmap, image):
    """Overlay heatmap on image."""
    out = np.float32(heatmap) / 255 + np.float32(image) / 255
    out = out / np.max(out)
    return np.uint8(255 * out)


class MVTecDataset(Dataset):
    """Custom dataset for MVTec Anomaly Detection."""

    def __init__(self, root, transform, gt_transform, phase='train'):
        self.phase = phase
        self.img_path = os.path.join(root, 'train' if phase == 'train' else 'test')
        self.gt_path = os.path.join(root, 'ground_truth') if phase != 'train' else None

        self.transform = transform
        self.gt_transform = gt_transform

        self.img_paths, self.gt_paths, self.labels, self.types = self._load_dataset()

    def _load_dataset(self):
        img_tot_paths, gt_tot_paths, tot_labels, tot_types = [], [], [], []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob(os.path.join(self.img_path, defect_type, "*.png"))
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob(os.path.join(self.img_path, defect_type, "*.png"))
                gt_paths = glob(os.path.join(self.gt_path, defect_type, "*.png")) if self.gt_path else []

                img_paths.sort()
                gt_paths.sort()

                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        label = self.labels[idx]
        img_type = self.types[idx]

        # Load and transform the image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # Load and transform the ground truth
        if self.phase == 'train':
            gt = torch.zeros_like(img[:1])  # No ground truth in the training phase
        else:
            if gt_path != 0:
                gt = Image.open(gt_path).convert('L')  # Load ground truth as grayscale
                gt = self.gt_transform(gt)
            else:
                gt = torch.zeros_like(img[:1])  # Create a blank tensor if no ground truth

        return img, gt, label, os.path.basename(img_path[:-4]), img_type


class UNet(nn.Module):
    """Customed UNET architecture that has a specific expanding path"""
    def __init__(self, in_channels):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(in_channels, in_channels // 2)
        self.enc2 = self.conv_block(in_channels // 2, in_channels // 4)
        self.enc3 = self.conv_block(in_channels // 4, in_channels // 8)

        # Bottleneck
        self.bottleneck = self.conv_block(in_channels // 8, in_channels // 8)
        
        # Output layer
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Convolutional block: Conv2d -> ReLU -> Conv2d -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding Path
        e1 = self.enc1(x)  # Downsample 1
        e2 = self.enc2(self.downsample(e1))  # Downsample 2
        e3 = self.enc3(self.downsample(e2))  # Downsample 3

        # Bottleneck
        b = self.bottleneck(self.downsample(e3))

        # Decoding Path
        d3 = self.upsample(b, e3)  # Upsample + Skip Connection from e3
        d2 = self.upsample(d3, e2)  # Upsample + Skip Connection from e2
        d1 = self.upsample(d2, e1)  # Upsample + Skip Connection from e1

        # Final Output
        out = self.out_conv(d1)
        return out

    def downsample(self, x):
        """Downsampling using MaxPool2d"""
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

    def upsample(self, x, skip_connection):
        """Upsampling with concatenation of skip connection"""
        x = nn.ConvTranspose2d(x.size(1), x.size(1), kernel_size=2, stride=2).to(x.device)(x)
        return torch.cat((x, skip_connection), dim=1)


class AnomalyDetector(nn.Module):
    """Student-Teacher Anomaly Detection Model."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Teacher model (frozen)
        self.model_t = models.resnet18(pretrained=True)
        self.model_t.eval()
        for param in self.model_t.parameters():
            param.requires_grad = False

        # Student model
        self.model_s = models.resnet18(pretrained=False)

        # Feature hooks
        self.features_t = []
        self.features_s = []

        # Register hooks for specific layers
        hooks = [
            self.model_t.layer1[-1].register_forward_hook(self._hook_t),
            self.model_t.layer2[-1].register_forward_hook(self._hook_t),
            self.model_t.layer3[-1].register_forward_hook(self._hook_t),
            self.model_s.layer1[-1].register_forward_hook(self._hook_s),
            self.model_s.layer2[-1].register_forward_hook(self._hook_s),
            self.model_s.layer3[-1].register_forward_hook(self._hook_s)
        ]

        # Define the three autoencoders for our different layers
        self.autoencoders = nn.ModuleList([
            UNet(64),   # The one for layer 1 waiting for feature maps with 64 channels
            UNet(128),  # The one for layer 2 waiting for feature maps with 128 channels
            UNet(256)   # The one for layer 3 waiting for feature maps with 256 channels
        ])

        self.criterion = nn.MSELoss(reduction='sum')
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )

    def _hook_t(self, module, input, output):
        self.features_t.append(output)

    def _hook_s(self, module, input, output):
        self.features_s.append(output)

    def forward(self, x):
        self.features_t.clear()
        self.features_s.clear()

        self.model_t(x)
        self.model_s(x)

        return self.features_t, self.features_s

    def get_teacher_features(self, x):
        """Passes data to the teacher to capture features."""
        self.features_t.clear()  # Reset before each run
        _ = self.model_t(x)  # Forward pass
        return self.features_t

    def cal_loss(self, features_s, features_t, epoch = 0, max_epochs=100):
        """Calculate combined loss: reconstruction + similarity."""
        tot_loss = 0
        alpha = 1e-7  # Fixed weight for reconstruction loss
        for i in range(len(features_t)):
            fs, ft = features_s[i], features_t[i]
            _, _, h, w = fs.shape

            # Reconstruction loss (autoencoder trying to reconstruct student features)
            reconstructed_fs = self.autoencoders[i](fs)
            reconstruction_loss = self.criterion(reconstructed_fs, fs)

            # Similarity loss (between reconstructed student and teacher features)
            reconstructed_fs_norm = F.normalize(reconstructed_fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            similarity_loss = (0.5 / (w * h)) * self.criterion(reconstructed_fs_norm, ft_norm)
            #print(f"Reconstruction loss: {alpha * reconstruction_loss}, Similarity loss: {similarity_loss}")
            # Combine both losses
            tot_loss += alpha * reconstruction_loss + similarity_loss

        return tot_loss

    def cal_anomaly_map(self, features_s, features_t, out_size=224):
        """Calculate anomaly map."""
        anomaly_map = np.ones([out_size, out_size]) if self.args.amap_mode == 'mul' else np.zeros([out_size, out_size])
        a_map_list = []

        for i in range(len(features_t)):
            fs, ft = features_s[i], features_t[i]
            fs_norm = F.normalize(fs, p=2)
            ft_norm = F.normalize(ft, p=2)
            a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
            a_map = a_map[0, 0, :, :].cpu().detach().numpy()
            a_map_list.append(a_map)

            anomaly_map = anomaly_map * a_map if self.args.amap_mode == 'mul' else anomaly_map + a_map

        return anomaly_map, a_map_list

# Retrieves the weights and biases of all autoencoders in a single dictionary using autoencoders.state_dict().
def save_autoencoders(autoencoders, save_path):
    """Saves autoencoder weights."""
    torch.save(autoencoders.state_dict(), save_path)
    print(f"Autoencoders saved to {save_path}")

# Reloads the weight dictionary from the file. Each sub-module (each autoencoder in the nn.ModuleList) finds its weights thanks to the association of state_dict keys.
def load_autoencoders(autoencoders, load_path):
    """Loads the autoencoder weights."""
    if os.path.exists(load_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoders.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Autoencoders loaded from {load_path}")
    else:
        print(f"No checkpoint found at {load_path}")

# This method is here to pretrain the autoencoder based on the output of the teacher. 
def pretrain_autoencoders(autoencoders, anomaly_detector, train_loader, device, save_path, mode = 'train'):
    """Pre-train the AE if wanted using Adam and MSE loss, careful, number of epoch have to be manually modified."""

    autoencoders.to(device)

    # Avoid pre-training and use the saved architecture
    if mode == 'load':
        load_autoencoders(autoencoders, save_path)
        return

    # If train :
    # Basic stuff nothing has been optimized
    optimizer = optim.Adam(autoencoders.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # teacher in eval mode cus we wont update it
    anomaly_detector.eval()
    anomaly_detector.model_t.eval()
    
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            x, _, _, _, _ = batch
            x = x.to(device)

            # Recovers teacher features via AnomalyDetector
            features_t = anomaly_detector.get_teacher_features(x)

            # Train each auto-encoder
            optimizer.zero_grad()
            loss = 0
            for i, feature_t in enumerate(features_t):
                feature_t = feature_t.detach()  # Pas de gradients pour le teacher
                reconstructed = autoencoders[i](feature_t)
                loss += criterion(reconstructed, feature_t)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/100], Loss: {total_loss:.4f}")

    # Save the architecture
    save_autoencoders(autoencoders, save_path)

from collections import defaultdict

def test_autoencoders(autoencoders, anomaly_detector, test_loader, device, save_dir="reconstructions", max_images_per_batch=5):
    """
    Test the autoencoders by reconstructing the features from the teacher model,
    saving the original and reconstructed features, and calculating MSE Loss per type.
    """
    os.makedirs(save_dir, exist_ok=True)
    anomaly_detector.eval()
    
     # Store MSEs by image type
    mse_per_type = defaultdict(list) 

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x, _, _, file_name, x_type = batch
            x = x.to(device)

            # Create subdirectories for each type (e.g., 'good', 'defect_name')
            type_dir = os.path.join(save_dir, x_type[0])
            os.makedirs(type_dir, exist_ok=True)

            # Get teacher features
            features_t = anomaly_detector.get_teacher_features(x)

            # Limit to max_images_per_batch
            num_images = min(x.size(0), max_images_per_batch)

            for img_idx in range(num_images):
                for i, feature_t in enumerate(features_t):
                    single_feature_t = feature_t[img_idx:img_idx + 1]  # [1, C, H, W]
                    reconstructed = autoencoders[i](single_feature_t)
                    
                    # Calculate MSE Loss for the current image
                    mse_loss = F.mse_loss(reconstructed, single_feature_t, reduction='mean').item()
                    mse_per_type[x_type[0]].append(mse_loss)  # Add MSE to the correct type

                    # Convert to numpy for visualization
                    recon_img = reconstructed.cpu().numpy().squeeze()
                    orig_img = single_feature_t.cpu().numpy().squeeze()

                    # If there are multiple channels, average them
                    if recon_img.ndim == 3:
                        recon_img = np.mean(recon_img, axis=0)
                        orig_img = np.mean(orig_img, axis=0)

                    # Normalize to [0, 255] for saving
                    recon_img = (min_max_norm(recon_img) * 255).astype(np.uint8)
                    orig_img = (min_max_norm(orig_img) * 255).astype(np.uint8)

                    recon_filename = os.path.join(type_dir, f"layer{i}_batch{batch_idx}_img{img_idx}_reconstructed.png")
                    orig_filename = os.path.join(type_dir, f"layer{i}_batch{batch_idx}_img{img_idx}_original.png")

                    cv2.imwrite(recon_filename, recon_img)
                    cv2.imwrite(orig_filename, orig_img)

    # Calculate and print average MSE per type
    print("Average MSE Loss per type:")
    for img_type, mse_list in mse_per_type.items():
        avg_mse = sum(mse_list) / len(mse_list)
        print(f"{img_type}: {avg_mse:.6f}")

def train(model, train_loader, optimizer, device):
    """Training loop."""
    model.train()
    total_loss = 0

    for batch in train_loader:
        x, _, _, _, _ = batch
        x = x.to(device)

        optimizer.zero_grad()
        features_t, features_s = model(x)
        loss = model.cal_loss(features_s, features_t)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def test(model, test_loader, device, sample_path):
    """Testing loop with anomaly map generation."""
    model.eval()
    gt_list_px_lvl, pred_list_px_lvl = [], []
    gt_list_img_lvl, pred_list_img_lvl = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, gt, label, file_name, x_type = batch
            x, gt, label = x.to(device), gt.to(device), label.to(device)

            features_t, features_s = model(x)

            # Pass features_s through autoencoders
            processed_features_s = []
            for i, feature_s in enumerate(features_s):
                processed_feature = model.autoencoders[i](feature_s)
                processed_features_s.append(processed_feature)

            # Calculate anomaly map using reconstructed features
            anomaly_map, a_map_list = model.cal_anomaly_map(features_s, features_t, out_size=args.input_size)

            # Pixel-level metrics
            gt_np = gt.cpu().numpy().astype(int)
            gt_list_px_lvl.extend(gt_np.ravel())
            pred_list_px_lvl.extend(anomaly_map.ravel())

            # Image-level metrics
            gt_list_img_lvl.append(label.cpu().numpy()[0])
            pred_list_img_lvl.append(anomaly_map.max())

            # Save anomaly maps
            x_inv = model.inv_normalize(x)
            input_x = cv2.cvtColor(x_inv.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            save_anomaly_maps(anomaly_map, a_map_list, input_x, gt_np[0][0] * 255, file_name[0], x_type[0], sample_path)

    # Compute AUC scores
    pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)

    return pixel_auc, img_auc


def save_anomaly_maps(anomaly_map, a_maps, input_x, gt_img, file_name, x_type, sample_path):
    """Save various anomaly map visualizations."""
    # Normalization and heatmap conversion
    anomaly_map_norm = min_max_norm(anomaly_map)
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm * 255)

    # Multi-scale anomaly maps
    am_maps = [min_max_norm(m) for m in a_maps]
    am_hmaps = [cvt2heatmap(m * 255) for m in am_maps]

    # Heatmap on original image
    heatmap = cvt2heatmap(anomaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap, input_x)

    # Save images
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}.jpg'), input_x)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)

    # Save multi-scale anomaly maps
    for i, am_hmap in enumerate(am_hmaps):
        cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_am{2 ** (i + 4)}.jpg'), am_hmap)


def main(args):
    # Set random seed for reproducibility
    set_seed()

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    print("Preprocess the data ...")
    data_transforms = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size), Image.LANCZOS),
        transforms.ToTensor(),
        transforms.CenterCrop(args.input_size), # Default is 256 so nothing happen
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Preprocess the ground truth ...")
    gt_transforms = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size), Image.LANCZOS),
        transforms.ToTensor(),
        transforms.CenterCrop(args.input_size)
    ])

    # Dataloaders
    train_dataset = MVTecDataset(
        root=os.path.join(args.dataset_path, args.category),
        transform=data_transforms,
        gt_transform=gt_transforms,
        phase='train'
    )
    print("Create the train loader ...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = MVTecDataset(
        root=os.path.join(args.dataset_path, args.category),
        transform=data_transforms,
        gt_transform=gt_transforms,
        phase='test'
    )
    print("Create the test loader ...")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model setup
    print("Set-up the model ...")
    model = AnomalyDetector(args).to(device)
    model.autoencoders.to(device)
    optimizer = optim.SGD([
        {'params': model.model_s.parameters(), 'lr': args.lr},
        {'params': model.autoencoders.parameters(), 'lr': args.lr / 10}
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    # Prepare directories
    print("Prepare directories ...")
    sample_path, _ = prep_dirs(os.path.join(args.project_path, args.category))
    autoencoder_save_path = os.path.join(args.project_path, args.pretrain_AE_name) # Path to the pretrained autoencoders

    if args.phase == 'train':
        # Pretrain the autoencoder (train or load)
        print("Pretrain the autoencoders ...")
        pretrain_autoencoders(
            model.autoencoders, model, train_loader, device, autoencoder_save_path, mode=args.pretrain_AE_mode
        )

        # Test the autoencoder
        if args.test_pretrained_AE == 'yes':
            print("Test the autoencoders ...")
            test_autoencoders(model.autoencoders, model, test_loader, device, save_dir="reconstructions", max_images_per_batch=3)

    if args.phase == 'train':
        # Training
        print("Starting Training...")
        for epoch in range(args.num_epochs):
            
            if epoch < 5:  # Freeze the Autoencoders for the first 5 periods
                for param in model.autoencoders.parameters():
                    param.requires_grad = False
            else:  # Unlock Autoencoder gradients after 5 epochs
                for param in model.autoencoders.parameters():
                    param.requires_grad = True

            train_loss = train(model, train_loader, optimizer, device)
            print(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {train_loss:.4f}")
        
        # Sauvegarde du modèle étudiant
        student_save_path = os.path.join(args.project_path, args.category, "student_model.pth")
        torch.save(model.model_s.state_dict(), student_save_path)
        print(f"Student model saved in {student_save_path}")

        # Sauvegarde des autoencodeurs fine tuné
        autoencoder_save_path = os.path.join(args.project_path, args.category, "fine_tuned_autoencoders.pth")
        torch.save(model.autoencoders.state_dict(), autoencoder_save_path)
        print(f"Autoencoders saved in {autoencoder_save_path}")
    
        # Testing
        print("Starting Testing...")
        pixel_auc, img_auc = test(model, test_loader, device, sample_path)
        print("\nResults:")
        print(f"Pixel-level AUC: {pixel_auc:.4f}")
        print(f"Image-level AUC: {img_auc:.4f}")
    
    elif args.phase == 'test':
        # Loading the student model
        student_save_path = os.path.join(args.project_path, args.category, "student_model.pth")
        model.model_s.load_state_dict(torch.load(student_save_path, map_location=device))
        model.model_s.to(device)
        print(f"Student model loaded from {student_save_path}")

        # Loading autoencoders
        autoencoder_save_path = os.path.join(args.project_path, args.category, "fine_tuned_autoencoders.pth")
        model.autoencoders.load_state_dict(torch.load(autoencoder_save_path, map_location=device))
        model.autoencoders.to(device)
        print(f"Autoencoders loaded from {autoencoder_save_path}")

        print("Starting Testing...")
        pixel_auc, img_auc = test(model, test_loader, device, sample_path)
        print("\nResults:")
        print(f"Pixel-level AUC: {pixel_auc:.4f}")
        print(f"Image-level AUC: {img_auc:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'D:\Dataset\mvtec_anomaly_detection') #/tile') #'/home/changwoo/hdd/datasets/mvtec_anomaly_detection'
    parser.add_argument('--category', default='leather')
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.0001)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path', default=r'/result') #'/home/changwoo/hdd/project_results/STPM_lightning/210621') #210605') #
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--amap_mode', choices=['mul','sum'], default='mul')
    parser.add_argument('--val_freq', default=5)
    parser.add_argument('--weights_file_version', type=str, default=None)
    parser.add_argument('--pretrain_AE_mode', choices=['train', 'load'], default='train')
    parser.add_argument('--pretrain_AE_name', default='UNET_leather.pth')
    parser.add_argument('--test_pretrained_AE', choices=['yes', 'no'], default='no')
    # parser.add_argument('--weights_file_version', type=str, default='version_1')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse arguments
    args = get_args()

    try:
        # Run main training and testing process
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
