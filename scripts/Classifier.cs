using Godot;
using System;
using TorchSharp;

/**

TODO:

Take voxel data

Split into 3D cube patches

Linear projection of flattened patches

Learned positional embedding

Transformer encoder

MLP Head

Classification

*/

public partial class Classifier : Node3D {

    [Export] int num_classes = 2;
    [Export] int emb_dim = 64;
    [Export] int nhead = 8;
    [Export] int mlp_dim = 256;
    [Export] float dropout = 0.1f;
    [Export] int depth = 6;
    [Export] int voxel_size = 32;
    [Export] int patch_size = 4;

    private ViT3D vit;
    private torch.optim.Optimizer optimizer;
    
    public override void _Ready() {
        vit = new ViT3D(
            voxel_size: voxel_size,
            patch_size: patch_size,
            emb_dim: emb_dim,
            num_classes: num_classes,
            dropout: dropout
        );

        optimizer = torch.optim.Adam(vit.parameters(), lr: 0.01);



        var dummy = GenerateDummyVoxels();

        var pred_y = Classify(dummy);    
    }

    private torch.Tensor Classify(torch.Tensor x) => vit.forward(x);


    private torch.Tensor GenerateDummyVoxels() {
        return torch.randn(new long[] { voxel_size, voxel_size, voxel_size }).round();
    }

    private torch.Tensor GeneratePatches(torch.Tensor voxels) {
        var n = voxel_size / patch_size;

        var patches = torch.zeros(new long[] { n * n * n, patch_size * patch_size * patch_size });

        for (int x = 0; x < n; x ++) {
            for (int y = 0; y < n; y ++) {
                for (int z = 0; z < n; z ++) {

                    int ix = x * patch_size;
                    int iy = y * patch_size;
                    int iz = z * patch_size;

                    patches[x + y * n + z * n * n] = voxels.index(
                        torch.TensorIndex.Slice(ix, ix + patch_size),
                        torch.TensorIndex.Slice(iy, iy + patch_size),
                        torch.TensorIndex.Slice(iz, iz + patch_size)
                    ).flatten();
                }
            }
        }

        return patches;
    }

    public override void _Process(double delta) {
    }
}
