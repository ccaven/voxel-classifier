using Godot;
using TorchSharp;
using static TorchSharp.torch;

[GlobalClass]
public partial class VariationalAutoEncoder : Module {
    [Export]
    public Module Encoder;

    [Export]
    public Module ToMu;

    [Export]
    public Module ToLogVar;
    
    [Export]
    public Module Decoder;

    public nn.Module<Tensor, Tensor> Reparameterize;

    public override nn.Module<Tensor, Tensor> Build() {
        ToMu.Init();
        ToLogVar.Init();
        Encoder.Init();
        Decoder.Init();

        Reparameterize = new Lambda<Tensor>(x => {
            var mu = ToMu.module.forward(x);
            var logVar = ToLogVar.module.forward(x);

            var std = exp(.5f * logVar);
            var eps = randn_like(std);

            return mu;
        });

        return nn.Sequential(
            Encoder.module,
            Reparameterize,
            Decoder.module
        );
    }
}