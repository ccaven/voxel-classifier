using static TorchSharp.torch;

public class SelfAttention : nn.Module<Tensor, Tensor> {
    private TorchSharp.Modules.MultiheadAttention mha;

    public SelfAttention(int emb_dim, int heads=8, float dropout=0f) : base("SelfAttention") {
        mha = nn.MultiheadAttention(emb_dim, heads, dropout);
    }

    public override Tensor forward(Tensor input) {
        return mha.forward(input, input, input, null, false, null).Item1;
    }
}

