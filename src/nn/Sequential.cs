using Godot;
using System.Collections.Generic;
using static TorchSharp.torch;

[GlobalClass]
public partial class Sequential : Module {
    [Export]
    public Godot.Collections.Array<Module> modules;

    public override nn.Module<Tensor, Tensor> Build() {
        List<nn.Module<Tensor, Tensor>> moduleList = new();

        foreach (var res in modules) {
            moduleList.Add(res.Build());
        }

        return nn.Sequential(moduleList);
    }
}