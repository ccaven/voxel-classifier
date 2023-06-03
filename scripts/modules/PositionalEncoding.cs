using Godot;
using TorchSharp;
using TorchSharp.Modules;

public class PositionalEncoding : torch.nn.Module<torch.Tensor, torch.Tensor> {
    public PositionalEncoding(int seq_len, int emb_dim, float dropout) : base("PositionalEncoding") {
        positionalEmbedding = new Parameter(torch.randn(seq_len + 1, emb_dim));
        classToken = new Parameter(torch.randn(1, emb_dim));
        dropoutModule = torch.nn.Dropout(dropout);
    }

    public override torch.Tensor forward(torch.Tensor x) {
        GD.Print("Incoming Positional Encoding:", string.Join(", ", x.shape));

        using var xWithClassToken = torch.cat(new [] { classToken, x }, 0);

        GD.Print("Incoming Positional Encoding:", string.Join(", ", xWithClassToken));

        var xWithPositionalEncoding = dropoutModule.forward(xWithClassToken + positionalEmbedding);
    
        GD.Print("Incoming Positional Encoding:", string.Join(", ", xWithPositionalEncoding));

        return xWithPositionalEncoding;
    }

    private readonly Parameter positionalEmbedding;
    private readonly Parameter classToken;
    private readonly Dropout dropoutModule;
}