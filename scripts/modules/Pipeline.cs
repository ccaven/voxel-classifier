using System;
using static TorchSharp.torch;

public delegate Tensor TensorLambda(Tensor input);

public class Pipeline : nn.Module<Tensor, Tensor> {
    private readonly TensorLambda[] fns;

    public Pipeline(params TensorLambda[] fns) : base("Pipeline") {
        this.fns = fns;
    }

    public override Tensor forward(Tensor x) {
        foreach (var fn in fns) x = fn(x);
        return x;
    }
}