using Godot;
using static TorchSharp.torch;

public class ExtractPatches3D : nn.Module<Tensor, Tensor> {
    private readonly int patch_size;
    private readonly int start_dim;

    public ExtractPatches3D(int patch_size, int start_dim=2) : base("ExtractPatches3D") {
        this.patch_size = patch_size;
        this.start_dim = start_dim;
    }

    public override Tensor forward(Tensor input) {
        using var patches = input
            .unfold(start_dim + 0, patch_size, patch_size)
            .unfold(start_dim + 1, patch_size, patch_size)
            .unfold(start_dim + 2, patch_size, patch_size);
        
        GD.Print("Patch dimensions:", string.Join(", ", patches.shape));

        var flattened_patches = patches
            .flatten(start_dim, start_dim + 2)
            .flatten(-3, -1);
        
        GD.Print("Flattened dimensions:", string.Join(", ", flattened_patches.shape));

        return flattened_patches;
    }
}

