using Godot;
using TorchSharp;

public enum ActivationType {
    ReLU,
    GELU,
    Sigmoid,
    Tanh
}

[GlobalClass]
public partial class Activation : Module {

    [Export]
    public ActivationType Type;

    public override torch.nn.Module<torch.Tensor, torch.Tensor> Build() => Type switch {
        ActivationType.ReLU => torch.nn.ReLU(),
        ActivationType.GELU => torch.nn.GELU(),
        ActivationType.Sigmoid => torch.nn.Sigmoid(),
        ActivationType.Tanh => torch.nn.Tanh(),
        _ => torch.nn.Identity()
    };
}