using static TorchSharp.torch;

public class Residual<T> : nn.Module<Tensor, Tensor> where T : nn.Module<Tensor, Tensor>{

    private readonly T inner;

    public Residual(T module) : base("Residual") {
        inner = module;
    }

    public override Tensor forward(Tensor input) {
        return input + inner.forward(input);
    }
}