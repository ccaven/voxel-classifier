using System;
using static TorchSharp.torch;

public class Transformer : nn.Module<Tensor, Tensor> {
    private nn.Module<Tensor, Tensor> net;

    public Transformer(int emb_dim, int depth, int heads, int mlp_dim, float dropout = 0f) : base("Transformer") {

        net = nn.Sequential();

        for (var i = 0; i < depth; i ++) {

            net.add_module((i*2).ToString(), 
                new Residual<PreNorm<SelfAttention>>(
                    new (
                        emb_dim, 
                        new ( emb_dim, heads, dropout ) 
                    ) 
                )
            );

            net.add_module((i*2+1).ToString(),
                new Residual<PreNorm<FeedForward>>(
                    new (
                        emb_dim,
                        new (emb_dim, mlp_dim)
                    )
                )
            );

        }

    }

    public override Tensor forward(Tensor input) => net.forward(input);
}
