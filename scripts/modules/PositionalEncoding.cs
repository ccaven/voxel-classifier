using Godot;
using TorchSharp.Modules;

using static TorchSharp.torch;

public class PositionalEncoding : nn.Module<Tensor, Tensor> {
    public PositionalEncoding(int seq_len, int emb_dim, float dropout) : base("PositionalEncoding") {
        positionalEmbedding = new Parameter(randn(seq_len + 1, emb_dim));
        classToken = new Parameter(randn(1, emb_dim));
        dropoutModule = nn.Dropout(dropout);
    }

    public override Tensor forward(Tensor x) {
        GD.Print("Incoming Positional Encoding:", string.Join(", ", x.shape));

        using var xWithClassToken = cat(new [] { classToken, x }, 0);

        GD.Print("Incoming Positional Encoding:", string.Join(", ", xWithClassToken));

        var xWithPositionalEncoding = dropoutModule.forward(xWithClassToken + positionalEmbedding);
    
        GD.Print("Incoming Positional Encoding:", string.Join(", ", xWithPositionalEncoding));

        return xWithPositionalEncoding;
    }

    public static Tensor GetClassToken(Tensor x) {
        if (x.shape.Length == 2) return x[0];
        else return x.index(TensorIndex.Colon, 0);
    }

    private readonly Parameter positionalEmbedding;
    private readonly Parameter classToken;
    private readonly Dropout dropoutModule;
}