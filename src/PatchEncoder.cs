using Godot;
using System.Threading;
using static TorchSharp.torch;

[GlobalClass]
public partial class PatchEncoder : Node {

    public static Device device = device(cuda.is_available() ? "cuda" : "cpu");

    // [Export]
    // public Noise noise;

    [Export]
    public VariationalAutoEncoder Encoder;

    [Export] 
    private Visual visual;

    private optim.Optimizer optimizer;

    public override void _Ready() {
        
        Encoder.Init();

        optimizer = optim.SGD(Encoder.module.parameters(), learningRate: 0.001);

        var trainThread = new Thread(new ThreadStart(Train));
        trainThread.Start();
    }


    private void Train () {
        
        Thread.Sleep(1000);

        Encoder.Init();

        for (var i = 0; i < 500; i ++) {
            var x = randn(new long[] { 50, 2, 2, 2 }, device: GlobalDevice.device)
                .repeat(new long[] { 1, 2, 2, 2 })
                .flatten(start_dim: 1);
            var recon_x = Encoder.module.forward(x);

            var recon_loss = nn.functional.mse_loss(x, recon_x);
            optimizer.zero_grad();
            recon_loss.backward();
            optimizer.step();

            visual.AddLossValue(i, recon_loss.item<float>(), 0f);

            Thread.Sleep(10);
        }

    }

}
