import functools
import spconv.pytorch as spconv
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
from .blocks import MLP, ResidualBlock, UBlock
from tree_learn.util.train import cuda_cast, point_wise_loss

LOSS_MULTIPLIER_SEMANTIC = 50 # multiply semantic loss for similar magnitude with offset loss
LOSS_MULTIPLIER_GEOMETRY = 1.0 # auxiliary weight for geometric consistency of predicted tree centers

class TreeLearn(nn.Module):
    def __init__(self,
                 channels=32,
                 num_blocks=7,
                 kernel_size=3,
                 dim_coord=3,
                 dim_feat=1,
                 dim_color=3,
                 rgb_hidden_channels=16,
                 use_geometry_constraint=True,
                 use_boundary_push_constraint=True,
                 boundary_push_weight=0.1,
                 boundary_push_warmup_epochs=5,
                 boundary_push_xy_radius=0.6,
                 boundary_push_z_radius=2.0,
                 boundary_push_margin=0.3,
                 boundary_push_max_anchors=384,
                 boundary_push_max_negatives=6,
                 fixed_modules=[],
                 use_feats=True,
                 use_coords=False,
                 spatial_shape=None,
                 max_num_points_per_voxel=3,
                 voxel_size=0.1,
                 **kwargs):

        super().__init__()
        self.voxel_size = voxel_size
        self.fixed_modules = fixed_modules
        self.use_feats = use_feats
        self.use_coords = use_coords
        self.spatial_shape = spatial_shape
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.dim_color = dim_color
        self.use_geometry_constraint = use_geometry_constraint
        self.use_boundary_push_constraint = use_boundary_push_constraint
        self.boundary_push_weight = boundary_push_weight
        self.boundary_push_warmup_epochs = boundary_push_warmup_epochs
        self.boundary_push_xy_radius = boundary_push_xy_radius
        self.boundary_push_z_radius = boundary_push_z_radius
        self.boundary_push_margin = boundary_push_margin
        self.boundary_push_max_anchors = boundary_push_max_anchors
        self.boundary_push_max_negatives = boundary_push_max_negatives

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        
        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                dim_coord + dim_feat, channels, kernel_size=kernel_size, padding=1, bias=False, indice_key='subm1'))
        block_channels = [channels * (i + 1) for i in range(num_blocks)]
        self.unet = UBlock(block_channels, norm_fn, 2, ResidualBlock, kernel_size, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(norm_fn(channels), nn.ReLU())

        # Lightweight point-wise RGB encoder for late gating; no parallel sparse U-Net to avoid OOM.
        self.rgb_encoder = nn.Sequential(
            nn.Linear(dim_color, rgb_hidden_channels),
            nn.ReLU(),
            nn.Linear(rgb_hidden_channels, rgb_hidden_channels),
            nn.ReLU()
        )
        self.rgb_gate = nn.Linear(rgb_hidden_channels, channels)
        
        # head
        self.semantic_linear = MLP(channels, 2, norm_fn=norm_fn, num_layers=2)
        self.offset_linear = MLP(channels, 3, norm_fn=norm_fn, num_layers=2)
        self.init_weights()
        # Zero-init the gate weights and use a negative bias so sigmoid(mask) starts near 0,
        # keeping the residual fusion close to the converged geometry-only baseline at startup.
        nn.init.constant_(self.rgb_gate.weight, 0)
        nn.init.constant_(self.rgb_gate.bias, -6.0)

        # weight init
        for mod in fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MLP):
                m.init_weights()


    # manually set batchnorms in fixed modules to eval mode
    def train(self, mode=True):
        super().train(mode)
        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for m in mod.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()


    def forward(self, batch, return_loss):
        backbone_output, v2p_map = self.forward_backbone(**batch)
        output = self.forward_head(backbone_output, v2p_map, batch.get('colors'))
        if return_loss:
            output = self.get_loss(model_output=output, **batch)
        
        return output

    @cuda_cast
    def forward_backbone(self, coords, input_feats, batch_ids, batch_size, **kwargs):
        voxel_feats, voxel_coords, v2p_map, spatial_shape = voxelize(torch.hstack([coords, input_feats]), batch_ids, batch_size, self.voxel_size, self.use_coords, self.use_feats, max_num_points_per_voxel=self.max_num_points_per_voxel)
        if self.spatial_shape is not None:
            spatial_shape = torch.tensor(self.spatial_shape, device=voxel_coords.device)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        output = self.input_conv(input)

        output = self.unet(output)
        output = self.output_layer(output)
        return output, v2p_map
    

    def forward_head(self, backbone_output, v2p_map, colors=None):
        output = dict()
        backbone_feats = backbone_output.features[v2p_map]
        if colors is None:
            raise KeyError("Late RGB gating expects point-wise `colors` with shape [N, 3].")
        if colors.shape[1] != self.dim_color:
            raise ValueError(f"Expected colors with {self.dim_color} channels, got shape {tuple(colors.shape)}.")
        colors = colors.to(device=backbone_feats.device, dtype=backbone_feats.dtype)

        rgb_encoded_feats = self.rgb_encoder(colors)
        gate_mask = torch.sigmoid(self.rgb_gate(rgb_encoded_feats))
        # SparseTensor alignment: fuse only after mapping decoder voxel features back to point order via v2p_map.
        fused_feats = backbone_feats + backbone_feats * gate_mask

        output['backbone_feats'] = backbone_feats
        output['rgb_gate_mask'] = gate_mask
        output['fused_backbone_feats'] = fused_feats
        output['semantic_prediction_logits'] = self.semantic_linear(fused_feats)
        output['offset_predictions'] = self.offset_linear(fused_feats)
        return output


    @cuda_cast
    def get_loss(self, model_output, semantic_labels, offset_labels, masks_off, masks_sem, coords, batch_ids, instance_labels, current_epoch=None, **kwargs):
        loss_dict = dict()
        
        # Define variables
        semantic_prediction_logits = model_output['semantic_prediction_logits'].float()
        offset_predictions = model_output['offset_predictions'].float()
        
        # semantic and offset losses
        semantic_loss, offset_loss = point_wise_loss(
            semantic_prediction_logits,
            offset_predictions, 
            masks_sem, masks_off,
            semantic_labels, offset_labels
        )
        loss_dict['semantic_loss'] = semantic_loss * LOSS_MULTIPLIER_SEMANTIC
        loss_dict['offset_loss'] = offset_loss
        if self.use_geometry_constraint:
            geometry_loss = self.get_geometry_constraint_loss(
                coords=coords,
                offset_predictions=offset_predictions,
                instance_labels=instance_labels,
                batch_ids=batch_ids,
                masks_off=masks_off
            )
            loss_dict['geometry_loss'] = geometry_loss * LOSS_MULTIPLIER_GEOMETRY
        if self.use_boundary_push_constraint:
            boundary_push_loss = self.get_boundary_push_loss(
                coords=coords,
                offset_predictions=offset_predictions,
                instance_labels=instance_labels,
                batch_ids=batch_ids,
                masks_off=masks_off
            )
            boundary_push_weight = self.get_boundary_push_weight(current_epoch)
            loss_dict['boundary_push_loss'] = boundary_push_loss * boundary_push_weight

        # Sum all losses
        loss = sum(_value for _value in loss_dict.values())
        return loss, loss_dict


    def get_geometry_constraint_loss(self, coords, offset_predictions, instance_labels, batch_ids, masks_off):
        valid_mask = masks_off & (instance_labels >= 0)
        if valid_mask.sum() == 0:
            return 0 * offset_predictions.sum()

        predicted_centers = coords[valid_mask] + offset_predictions[valid_mask]
        instance_labels = instance_labels[valid_mask]
        batch_ids = batch_ids[valid_mask]
        batch_instance_ids = torch.stack([batch_ids, instance_labels], dim=1)
        unique_batch_instance_ids = torch.unique(batch_instance_ids, dim=0)

        geometry_losses = []
        for batch_instance_id in unique_batch_instance_ids:
            same_instance_mask = (batch_instance_ids == batch_instance_id).all(dim=1)
            instance_predicted_centers = predicted_centers[same_instance_mask]
            if len(instance_predicted_centers) <= 1:
                continue

            # Restore the geometric constraint by forcing points of the same tree
            # to vote for a compact common center after applying the predicted offsets.
            instance_center_mean = instance_predicted_centers.mean(dim=0, keepdim=True)
            instance_geometry_loss = (instance_predicted_centers - instance_center_mean).pow(2).sum(dim=1).sqrt().mean()
            geometry_losses.append(instance_geometry_loss)

        if not geometry_losses:
            return 0 * offset_predictions.sum()
        return torch.stack(geometry_losses).mean()


    def get_boundary_push_weight(self, current_epoch):
        if current_epoch is None or self.boundary_push_warmup_epochs <= 0:
            return self.boundary_push_weight

        warmup_progress = max(float(current_epoch) - 1.0, 0.0) / float(self.boundary_push_warmup_epochs)
        warmup_progress = min(warmup_progress, 1.0)
        return self.boundary_push_weight * warmup_progress


    def get_boundary_push_loss(self, coords, offset_predictions, instance_labels, batch_ids, masks_off):
        valid_mask = masks_off & (instance_labels >= 0)
        if valid_mask.sum() <= 1:
            return 0 * offset_predictions.sum()

        raw_coords = coords[valid_mask]
        shifted_coords = raw_coords + offset_predictions[valid_mask]
        instance_labels = instance_labels[valid_mask]
        batch_ids = batch_ids[valid_mask]

        sample_losses = []
        unique_batch_ids = torch.unique(batch_ids)
        for sample_batch_id in unique_batch_ids:
            sample_mask = batch_ids == sample_batch_id
            if sample_mask.sum() <= 1:
                continue

            sample_raw_xy = raw_coords[sample_mask, :2]
            sample_raw_z = raw_coords[sample_mask, 2]
            sample_shifted_xy = shifted_coords[sample_mask, :2]
            sample_instance_labels = instance_labels[sample_mask]
            num_points = len(sample_raw_xy)
            if num_points <= 1:
                continue

            if num_points > self.boundary_push_max_anchors:
                anchor_indices = torch.randperm(num_points, device=sample_raw_xy.device)[:self.boundary_push_max_anchors]
            else:
                anchor_indices = torch.arange(num_points, device=sample_raw_xy.device)

            anchor_raw_xy = sample_raw_xy[anchor_indices]
            anchor_raw_z = sample_raw_z[anchor_indices]
            anchor_shifted_xy = sample_shifted_xy[anchor_indices]
            anchor_instance_labels = sample_instance_labels[anchor_indices]

            raw_xy_distances = torch.cdist(anchor_raw_xy, sample_raw_xy)
            raw_z_distances = torch.abs(anchor_raw_z[:, None] - sample_raw_z[None, :])
            different_instance_mask = anchor_instance_labels[:, None] != sample_instance_labels[None, :]
            boundary_neighbor_mask = (
                different_instance_mask
                & (raw_xy_distances < self.boundary_push_xy_radius)
                & (raw_z_distances < self.boundary_push_z_radius)
            )
            if not boundary_neighbor_mask.any():
                continue

            shifted_xy_distances = torch.cdist(anchor_shifted_xy, sample_shifted_xy)
            for anchor_row in range(len(anchor_indices)):
                negative_indices = torch.nonzero(boundary_neighbor_mask[anchor_row], as_tuple=False).flatten()
                if len(negative_indices) == 0:
                    continue

                if len(negative_indices) > self.boundary_push_max_negatives:
                    negative_raw_distances = raw_xy_distances[anchor_row, negative_indices]
                    _, nearest_negative_order = torch.topk(
                        negative_raw_distances,
                        k=self.boundary_push_max_negatives,
                        largest=False
                    )
                    negative_indices = negative_indices[nearest_negative_order]

                push_distances = shifted_xy_distances[anchor_row, negative_indices]
                push_margin_violations = torch.relu(self.boundary_push_margin - push_distances)
                if push_margin_violations.numel() > 0:
                    sample_losses.append(push_margin_violations.pow(2).mean())

        if not sample_losses:
            return 0 * offset_predictions.sum()
        return torch.stack(sample_losses).mean()


def voxelize(feats, batch_ids, batch_size, voxel_size, use_coords, use_feats, max_num_points_per_voxel, epsilon=1):
    voxel_coords, voxel_feats, v2p_maps = [], [], []
    total_len_voxels = 0
    for i in range(batch_size):
        feats_one_element = feats[batch_ids == i]
        min_range = torch.min(feats_one_element[:, :3], dim=0).values
        max_range = torch.max(feats_one_element[:, :3], dim=0).values + epsilon
        voxelizer = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, voxel_size], 
            coors_range_xyz=min_range.tolist() + max_range.tolist(),
            num_point_features=feats.shape[1], 
            max_num_voxels=len(feats), 
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=feats.device)
        voxel_feat, voxel_coord, _, v2p_map = voxelizer.generate_voxel_with_id(feats_one_element)
        assert torch.sum(v2p_map == -1) == 0
        voxel_coord[:, [0, 2]] = voxel_coord[:, [2, 0]]
        voxel_coord = torch.cat((torch.ones((len(voxel_coord), 1), device=feats.device)*i, voxel_coord), dim=1)

        # get mean feature of voxel
        zero_rows = torch.sum(voxel_feat == 0, dim=2) == voxel_feat.shape[2]
        voxel_feat[zero_rows] = float("nan")
        voxel_feat = torch.nanmean(voxel_feat, dim=1)
        if not use_coords:
            voxel_feat[:, :3] = torch.ones_like(voxel_feat[:, :3])
        if not use_feats:
            voxel_feat[:, 3:] = torch.ones_like(voxel_feat[:, 3:])
        voxel_feat = torch.hstack([voxel_feat[:, 3:], voxel_feat[:, :3]])

        voxel_coords.append(voxel_coord)
        voxel_feats.append(voxel_feat)
        v2p_maps.append(v2p_map + total_len_voxels)
        total_len_voxels += len(voxel_coord) 
    voxel_coords = torch.cat(voxel_coords, dim=0)
    voxel_feats = torch.cat(voxel_feats, dim=0)
    v2p_maps = torch.cat(v2p_maps, dim=0)
    spatial_shape = voxel_coords.max(dim=0).values + 1

    return voxel_feats, voxel_coords, v2p_maps, spatial_shape[1:]
