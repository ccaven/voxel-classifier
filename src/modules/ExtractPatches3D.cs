using static TorchSharp.torch;

public class ExtractPatches3D : nn.Module<Tensor, Tensor> {
    private readonly int patch_size;
    private readonly int start_dim;

    public ExtractPatches3D(int patch_size, int start_dim=2) : base("ExtractPatches3D") {
        this.patch_size = patch_size;
        this.start_dim = start_dim;
    }

    public override Tensor forward(Tensor input) {
        return input
            .unfold(start_dim + 0, patch_size, patch_size)
            .unfold(start_dim + 1, patch_size, patch_size)
            .unfold(start_dim + 2, patch_size, patch_size)
            .flatten(start_dim, start_dim + 2)
            .flatten(-3, -1);
    }
}