import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader # Assume DataLoader provides batches like {"noisy": ..., "clean": ...}
from tqdm import tqdm
import logging
import os

# --- Assume these are defined elsewhere ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_loader = DataLoader(...) # Your training data loader
# val_loader = DataLoader(...)   # Your validation data loader
# OUTPUT_DIR = "./output" # Example output directory
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)
# -----------------------------------------

# --- 1. Denoising Model Architecture (CNN) ---
class DenoiserCNN(nn.Module):
    """Simple CNN (U-Net like structure) for denoising 2D Wigner functions."""
    def __init__(self, n_channels=1):
        super(DenoiserCNN, self).__init__()
        # Encoder Path
        self.enc1 = self._conv_block(n_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(64, 128) # Bottleneck layer

        # Decoder Path
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input channels to dec2 = channels from upsampled layer + channels from skip connection (e2)
        self.dec2 = self._conv_block(128 + 64, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input channels to dec1 = channels from upsampled layer + channels from skip connection (e1)
        self.dec1 = self._conv_block(64 + 32, 32)

        # Final Convolution layer to map back to original number of channels
        self.final_conv = nn.Conv2d(32, n_channels, kernel_size=1) # 1x1 conv

    def _conv_block(self, in_channels, out_channels):
        """Helper function for a standard convolutional block with BatchNorm."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x) # Output shape depends on input, e.g., (B, 32, H, W)
        p1 = self.pool1(e1) # (B, 32, H/2, W/2)
        e2 = self.enc2(p1) # (B, 64, H/2, W/2)
        p2 = self.pool2(e2) # (B, 64, H/4, W/4)
        e3 = self.enc3(p2) # (B, 128, H/4, W/4) Bottleneck

        # Decoder with Skip Connections
        d2 = self.up2(e3) # (B, 128, H/2, W/2)
        # Concatenate skip connection from encoder (e2)
        d2 = torch.cat([d2, e2], dim=1) # (B, 128 + 64, H/2, W/2)
        d2 = self.dec2(d2) # (B, 64, H/2, W/2)

        d1 = self.up1(d2) # (B, 64, H, W)
        # Concatenate skip connection from encoder (e1)
        d1 = torch.cat([d1, e1], dim=1) # (B, 64 + 32, H, W)
        d1 = self.dec1(d1) # (B, 32, H, W)

        # Output
        out = self.final_conv(d1) # (B, n_channels, H, W)
        return out


# --- 2. Loss Function Definitions ---

# Standard Loss (Example: Mean Squared Error)
standard_loss_fn = nn.MSELoss()

# --- CUSTOM LOSS FUNCTION IMPLEMENTATION ---
def custom_loss_function(predictions, targets, noisy_inputs=None, model_params=None, negativity_weight=0.1):
    """
    Calculates a custom loss, combining MSE with a Wigner negativity penalty.

    Args:
        predictions (torch.Tensor): Model output (denoised Wigners) [B, C, H, W].
        targets (torch.Tensor): Ground truth (clean Wigners) [B, C, H, W].
        noisy_inputs (torch.Tensor, optional): Original noisy input Wigners.
        model_params (iterable, optional): Model parameters for regularization.
        negativity_weight (float): Weighting factor for the negativity penalty term.

    Returns:
        torch.Tensor: A scalar loss value.
    """
    # 1. Standard MSE Loss component
    mse_loss = standard_loss_fn(predictions, targets)

    # 2. Custom Negativity Penalty component (Example)
    # Flatten spatial dimensions to find min value per sample in batch
    # Assumes channel dimension C=1
    pred_flat = predictions.view(predictions.size(0), -1) # Shape: [B, H*W]
    target_flat = targets.view(targets.size(0), -1)     # Shape: [B, H*W]

    pred_min = torch.min(pred_flat, dim=1)[0]      # Shape: [B]
    target_min = torch.min(target_flat, dim=1)[0]  # Shape: [B]

    # Penalize if prediction's minimum is less negative (i.e., higher value)
    # than the target's minimum, especially when the target is negative.
    negativity_mismatch = torch.relu(pred_min - target_min) # Only positive differences penalized

    # Optional: Weight penalty more if the target actually had negativity
    is_target_negative_mask = (target_min < -1e-6).float() # Mask for targets with non-trivial negativity

    # Calculate the average penalty across the batch
    # Weighted by the mask, so only samples where target was negative contribute strongly
    negativity_penalty = torch.mean(negativity_mismatch * is_target_negative_mask)

    # 3. Combine the losses
    total_loss = mse_loss + negativity_weight * negativity_penalty

    # --- Add other custom loss components here if needed ---
    # For example, regularization on model parameters:
    # l1_reg = 0.0
    # l2_reg = 0.0
    # if model_params is not None:
    #     l1_lambda = 1e-5
    #     l2_lambda = 1e-4
    #     for param in model_params:
    #         l1_reg += torch.norm(param, 1)
    #         l2_reg += torch.norm(param, 2)**2
    #     total_loss += l1_lambda * l1_reg + l2_lambda * l2_reg

    return total_loss

# --- 3. Training Loop ---
def train_model_with_custom_loss(model, train_loader, val_loader, optimizer, num_epochs, device, use_custom_loss=True):
    """Trains the denoising model using the specified loss function."""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # Determine which loss function to use
    if use_custom_loss:
        loss_fn = custom_loss_function # Use the custom one
        logger.info("Using CUSTOM loss function for training.")
    else:
        loss_fn = standard_loss_fn # Use the standard MSE
        logger.info("Using STANDARD MSE loss function for training.")

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in train_pbar:
            # Ensure data is on the correct device
            noisy_wigners = batch["noisy"].to(device)
            clean_wigners = batch["clean"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Get model predictions
            outputs = model(noisy_wigners)

            # Calculate loss using the chosen loss function
            # Pass necessary arguments to the custom loss if using it
            if use_custom_loss:
                 # Example: pass model parameters if regularization is used in custom loss
                 # loss = loss_fn(outputs, clean_wigners, noisy_inputs=noisy_wigners, model_params=model.parameters())
                 loss = loss_fn(outputs, clean_wigners, noisy_inputs=noisy_wigners) # Simpler call without regularization
            else:
                 loss = loss_fn(outputs, clean_wigners)

            # Backward pass: Compute gradient of the loss w.r.t. model parameters
            loss.backward()

            # Optimize: Update model parameters based on gradients
            optimizer.step()

            running_train_loss += loss.item()
            # Update progress bar description with current batch loss
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        running_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad(): # Disable gradient calculations for validation
            for batch in val_pbar:
                noisy_wigners = batch["noisy"].to(device)
                clean_wigners = batch["clean"].to(device)

                outputs = model(noisy_wigners)

                # Calculate validation loss using the same loss function
                if use_custom_loss:
                     val_loss = loss_fn(outputs, clean_wigners, noisy_inputs=noisy_wigners)
                else:
                     val_loss = loss_fn(outputs, clean_wigners)


                running_val_loss += val_loss.item()
                val_pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        logger.info(f"Epoch {epoch+1}/{num_epochs} Summary - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the model checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(OUTPUT_DIR, "best_denoiser_model.pth")
            try:
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Validation loss decreased to {best_val_loss:.4f}. Saved model checkpoint to {model_save_path}")
            except Exception as e:
                logger.error(f"Failed to save model checkpoint: {e}")


    logger.info("Training finished.")
    # Return the trained model and loss history for further analysis/use
    return model, train_losses, val_losses

# --- Example Usage ---
# Need to instantiate model, optimizer, and dataloaders first

# model = DenoiserCNN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# num_epochs = 20

# trained_model, train_history, val_history = train_model_with_custom_loss(
#     model=model,
#     train_loader=train_loader, # Assumed to be defined
#     val_loader=val_loader,     # Assumed to be defined
#     optimizer=optimizer,
#     num_epochs=num_epochs,
#     device=device,
#     use_custom_loss=True # Set to True to use custom_loss_function, False for standard_loss_fn
# )


