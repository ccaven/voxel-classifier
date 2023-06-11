using TorchSharp;
using static TorchSharp.torch;

public delegate T LambdaDelegate<T> (Tensor input);

public class Lambda<T> : nn.Module<Tensor, T> {

    private readonly LambdaDelegate<T> fn;

    public Lambda(LambdaDelegate<T> fn) : base("SplitInto") {
        this.fn = fn;
    }

    public override T forward(Tensor input) => fn(input);
}
