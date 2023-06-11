using Godot;
using TorchSharp.Modules;

using static TorchSharp.torch;

public class PositionalEncoding : nn.Module<Tensor, Tensor> {
    public PositionalEncoding(int seq_len, int emb_dim, float dropout) : base("PositionalEncoding") {
        positionalEmbedding = new Parameter(randn(1, seq_len + 1, emb_dim));
        classToken = new Parameter(randn(1, 1, emb_dim));
        dropoutModule = nn.Dropout(dropout);
    }

    public override Tensor forward(Tensor x) {
        return dropoutModule.forward(
            cat(new [] { 
                classToken.repeat(x.shape[0], 1, 1),
                x
            }, 
            1
        ) + positionalEmbedding);
    }

    private readonly Parameter positionalEmbedding;
    private readonly Parameter classToken;
    private readonly Dropout dropoutModule;

    public static readonly Lambda<Tensor> toClassToken = new (x => x.index(TensorIndex.Colon, 0));
}