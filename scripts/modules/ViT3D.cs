using TorchSharp.Modules;
using static TorchSharp.torch;

public class ViT3D : nn.Module<Tensor, Tensor> {

    private readonly nn.Module<Tensor, Tensor> net;
    private readonly nn.Module<Tensor, Tensor> mlp;

    public ViT3D(int voxel_size, int patch_size, int emb_dim, int num_classes, float dropout = 0f) : base("Vit3D") {
        var seq_len = voxel_size / patch_size * voxel_size / patch_size * voxel_size / patch_size;
        var patch_dim = patch_size * patch_size * patch_size;

        var patching = new ExtractPatches3D(patch_size, start_dim: 0);

        var embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        );

        var encoding = new PositionalEncoding(seq_len, emb_dim, dropout);

        var transformer = new Transformer(emb_dim, 4, 8, emb_dim * 2, dropout);

        net = nn.Sequential(
            patching,
            embedding,
            encoding,
            transformer
        );

        mlp = new PreNorm<Linear>(emb_dim, nn.Linear(emb_dim, num_classes));
    }

    public override Tensor forward(Tensor input) {
        using var enc = net.forward(input);
        using var enc0 = enc[0];
        return mlp.forward(enc0);
    }
}
