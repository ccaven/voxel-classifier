using Godot;
using static TorchSharp.torch;

[GlobalClass]
public partial class ModuleNode : Node {

    [Export]
    public Module model;

    public override void _Ready() {
        var m = model.Build();

        var dummy = randn(new long[] { 10 });

        GD.Print(dummy);

        var res = m.forward(dummy);

        GD.Print(res);
    }

}

