using static TorchSharp.torch;

public class PreNorm<T> : nn.Module<Tensor, Tensor> where T : nn.Module<Tensor, Tensor>{

    private readonly T inner;
    private readonly TorchSharp.Modules.LayerNorm layerNorm;

    public PreNorm(int dim, T module) : base("PreNorm") {
        inner = module;
        layerNorm = nn.LayerNorm(dim);
    }

    public override Tensor forward(Tensor input) {
        return inner.forward(layerNorm.forward(input));
    }
}