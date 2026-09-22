import torch.nn as nn

from models.mano import NUM_BETAS, NUM_POSE_PCA, create_mano_layers, mano_joints


class Decoder(nn.Module):
    def __init__(self, mano_dir, latent_dim=256):
        super().__init__()
        self.mano_layer_left, self.mano_layer_right = create_mano_layers(mano_dir)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_POSE_PCA + NUM_BETAS + 3 + 3),
        )

    def decode_z(self, z, is_right):
        lead = z.shape[:-1]
        params = self.fc(z.reshape(-1, z.shape[-1]))
        pose = params[:, :NUM_POSE_PCA]
        betas = params[:, NUM_POSE_PCA:NUM_POSE_PCA + NUM_BETAS]
        rot = params[:, NUM_POSE_PCA + NUM_BETAS:NUM_POSE_PCA + NUM_BETAS + 3]
        trans = params[:, -3:]
        layer = self.mano_layer_right if is_right else self.mano_layer_left
        joints = mano_joints(layer, betas, pose, rot, trans)
        return joints.reshape(*lead, joints.shape[-2], 3)

    def forward(self, z_left, z_right):
        return self.decode_z(z_left, is_right=False), self.decode_z(z_right, is_right=True)
