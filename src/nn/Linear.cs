using Godot;
using TorchSharp;

[GlobalClass]
public partial class Linear : Module {
    [Export]
    public int InputDimensions;

    [Export]
    public int OutputDimensions;

    public override torch.nn.Module<torch.Tensor, torch.Tensor> Build() => torch.nn.Linear(InputDimensions, OutputDimensions, device: GlobalDevice.device);
}