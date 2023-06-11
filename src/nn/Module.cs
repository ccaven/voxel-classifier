using Godot;
using static TorchSharp.torch;

[GlobalClass]
public partial class Module : Resource {

    public nn.Module<Tensor, Tensor> module;

    public virtual nn.Module<Tensor, Tensor> Build() {
        throw new System.NotImplementedException();
    }

    public nn.Module<Tensor, Tensor> Init() {
        module = Build();
        return module;
    }
}