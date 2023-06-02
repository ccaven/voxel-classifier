using static TorchSharp.torch;

public class FeedForward : nn.Module<Tensor, Tensor> {
    private nn.Module<Tensor, Tensor> net;

    public FeedForward(int dim, int hidden_dim, float dropout = 0f) : base("FeedForward") {
        net = nn.Sequential(
            ("l-1", nn.Linear(dim, hidden_dim)),
            ("g-1", nn.GELU()),
            ("d-1", nn.Dropout(dropout)),
            ("l-2", nn.Linear(hidden_dim, dim)),
            ("d-2", nn.Dropout(dropout))
        );
    }

    public override Tensor forward(Tensor input) => net.forward(input);
}

