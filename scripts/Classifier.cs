using Godot;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;

public partial class Classifier : Node3D {

    [Export] int num_classes = 3;
    [Export] int emb_dim = 32;
    [Export] int nhead = 8;
    [Export] int mlp_dim = 256;
    [Export] float dropout = 0.1f;
    [Export] int depth = 6;
    [Export] int voxel_size = 32;
    [Export] int patch_size = 4;

    [Export] VoxelGenerator generator;

    private nn.Module<Tensor, Tensor> classifier;
    private optim.Optimizer optimizer;
    private Loss<Tensor, Tensor, Tensor> loss_fn;

    private Thread trainThread;

    public override void _Ready() {
        generator.SetSize(voxel_size);
        generator.SetNumClasses(num_classes);

        var patch_dim = patch_size * patch_size * patch_size;
        var seq_len = voxel_size / patch_size * voxel_size / patch_size * voxel_size / patch_size;

        classifier = nn.Sequential(
            new ExtractPatches3D(patch_size, start_dim: 1),
            nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, emb_dim),
                nn.LayerNorm(emb_dim)
            ),
            new PositionalEncoding(seq_len, emb_dim, dropout),
            new Transformer(emb_dim, 3, 8, 128, dropout),
            PositionalEncoding.toClassToken,
            nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, num_classes)
            )
        ).cuda();

        optimizer = optim.Adam(classifier.parameters(), lr: 0.01);

        loss_fn = nn.CrossEntropyLoss().cuda();

        trainThread = new Thread(new ThreadStart(TrainLoop));
        trainThread.Start();
    }

    private void TrainLoop () {
        for (var i = 0; i < 100; i ++) {
            var (x, y) = generator.GenerateRandomBatch(10);
            var pred_y = classifier.forward(x);

            var loss = loss_fn.forward(pred_y, y);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            GD.Print(loss.item<float>());
        }
    }

    public override void _Process(double delta) {
        


    }

    public void TrainBatch(Tensor voxels, Tensor labels) {
        var pred_labels = classifier.forward(voxels);
        var loss = loss_fn.forward(pred_labels, labels);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
