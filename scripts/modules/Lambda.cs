using static TorchSharp.torch;

public class Lambda : nn.Module<Tensor, Tensor> {
    private readonly TensorLambda fn;
    
    public Lambda(TensorLambda fn) : base("Lambda") {
        this.fn = fn;
    }

    public override Tensor forward(Tensor input) => fn(input);
}