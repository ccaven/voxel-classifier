using Godot;
using System;
using TorchSharp;
using static TorchSharp.torch;

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
    [Export] int emb_dim = 32;
    [Export] int nhead = 8;
    [Export] int mlp_dim = 256;
    [Export] float dropout = 0.1f;
    [Export] int depth = 6;
    [Export] int voxel_size = 32;
    [Export] int patch_size = 4;

    private ViT3D vit;
    private optim.Optimizer optimizer;
    
    public override void _Ready() {

        var dummy = GenerateDummyVoxels();

        var extractPatches = new ExtractPatches3D(patch_size, start_dim: 0);
        
        var patch_dim = patch_size * patch_size * patch_size;
        var embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        );

        var seq_len = voxel_size / patch_size * voxel_size / patch_size * voxel_size / patch_size;
        var encoding = new PositionalEncoding(seq_len, emb_dim, dropout);
        var transformer = new Transformer(emb_dim, 3, 8, 128, dropout);

        var mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes),
            nn.LayerNorm(num_classes)
        );

        var pipeline = new Pipeline(
            x => extractPatches.forward(x),
            x => embedding.forward(x),
            x => encoding.forward(x),
            x => transformer.forward(x),
            x => x[0],
            x => mlp.forward(x)
        );

        var result = pipeline.forward(dummy);

        GD.Print(result);
    }

    private Tensor Classify(Tensor x) => vit.forward(x);

    private Tensor GenerateDummyVoxels() {
        return randn(new long[] { voxel_size, voxel_size, voxel_size }).round();
    }

    private Tensor GeneratePatches(Tensor voxels) {
        var n = voxel_size / patch_size;

        var patches = zeros(new long[] { n * n * n, patch_size * patch_size * patch_size });

        for (int x = 0; x < n; x ++) {
            for (int y = 0; y < n; y ++) {
                for (int z = 0; z < n; z ++) {

                    int ix = x * patch_size;
                    int iy = y * patch_size;
                    int iz = z * patch_size;

                    patches[x + y * n + z * n * n] = voxels.index(
                        TensorIndex.Slice(ix, ix + patch_size),
                        TensorIndex.Slice(iy, iy + patch_size),
                        TensorIndex.Slice(iz, iz + patch_size)
                    ).flatten();
                }
            }
        }

        return patches;
    }

    public override void _Process(double delta) {
    }
}
