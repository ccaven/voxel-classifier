using Godot;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;

public partial class Classifier : Node3D {

    [Export] int num_classes = 4;
    [Export] int emb_dim = 32;
    [Export] int nhead = 4;
    [Export] int mlp_dim = 32;
    [Export] float dropout = 0.0f;
    [Export] int depth = 1;
    [Export] int voxel_size = 8;
    [Export] int patch_size = 2;

    [Export] private VoxelGenerator generator;
    [Export] private Visual visual;

    [Signal] public delegate void EpochFinishedEventHandler(long epoch, float loss);

    private nn.Module<Tensor, Tensor> classifier;
    private optim.Optimizer optimizer;
    private Loss<Tensor, Tensor, Tensor> loss_fn;
    private optim.lr_scheduler.LRScheduler scheduler;

    private Thread trainThread;

    public override void _Ready() {
        generator.SetSize(voxel_size);
        generator.SetPatchSize(patch_size);
        generator.SetNumClasses(num_classes);

        var patch_dim = patch_size * patch_size * patch_size;
        var seq_len = voxel_size / patch_size * voxel_size / patch_size * voxel_size / patch_size;

        /*
        classifier = nn.Sequential(
            nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, emb_dim),
                nn.LayerNorm(emb_dim)
            ),
            new PositionalEncoding(seq_len, emb_dim, dropout),
            PositionalEncoding.toClassToken,
            nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, num_classes),
                nn.Softmax(dim: 1)
            )
        ).cuda();
        */

        var embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        ).cuda();

        var positionalEncoding = new PositionalEncoding(seq_len, emb_dim, dropout).cuda();

        var toClassToken = PositionalEncoding.toClassToken.cuda();

        var transformerEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_dim, nhead, 256, dropout),
            depth
        ).cuda();

        var x_mask = zeros(new long[] { seq_len, seq_len }).cuda();

        var mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes),
            nn.Softmax(dim: 1)
        ).cuda();

        classifier = new Pipeline(
            x => embedding.forward(x),
            x => positionalEncoding.forward(x),
            x => transformerEncoder.forward(x, null, null),
            x => toClassToken.forward(x),
            x => mlp.forward(x)
        ).cuda();
        

        /*
        classifier = nn.Sequential(
            nn.Flatten(startDim: 1),
            nn.Linear(seq_len * patch_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim: 1)
        );
        */

        optimizer = optim.Adam(classifier.parameters(), lr: 0.005);
        // scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma: 0.01);

        loss_fn = nn.NLLLoss().cuda();

        trainThread = new Thread(new ThreadStart(TrainLoop));
        trainThread.Start();
    }

    private void TrainLoop () {
        Thread.Sleep(1000);
        
        int batchSize = 15;

        for (var i = 0; i < 50; i ++) {
            
            GD.Print("Starting epoch ", i);

            var (x, y) = generator.GenerateRandomBatch(batchSize, cuda: true);

            GD.Print("x: ", x);
            GD.Print("y: ", y);

            using var pred_y = classifier.forward(x);

            GD.Print("pred_y: ", pred_y);

            var loss = loss_fn.forward(pred_y, y);

            optimizer.zero_grad();
            
            loss.backward();
            
            optimizer.step();

            //scheduler.step();

            var correct = pred_y.argmax(dim: 1) == y;
            var accuracy = correct.to(ScalarType.Int32).sum(ScalarType.Int32).item<int>();

            visual.AddLossValue(i, loss.item<float>(), (float)accuracy / batchSize);

            // EmitSignal(SignalName.EpochFinished, i, loss.item<float>());
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
