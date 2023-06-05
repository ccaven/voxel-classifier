using Godot;
using System;

using static TorchSharp.torch;

delegate float VoxelLambda(Tensor position);

public partial class VoxelGenerator : Node3D {

    [Export] int size;
    [Export] int num_classes;

    public VoxelGenerator() : base() {}

    public int SetSize(int size) => this.size = size;

    public int SetNumClasses(int num_classes) => this.num_classes = num_classes;

    public Tensor GenerateWorley(int n) {
        var center = randn(n, 3) * size;
        var radius = randn(n) * (size * 0.5f);

        return ApplyLambda(p => {
            var dp = center - p;
            var d = linalg.norm(dp, 2, dims: new long[] { 1 }) - radius;
            return d.max().item<float>() > 0 ? 1 : 0;
        });
    }

    public Tensor GenerateRandom() => randn(size, size, size).round();
    
    public Tensor GeneratePlane(Vector3 normal) {
        var normal_tensor = tensor(new float[] { normal.X, normal.Y, normal.Z });
        
        return ApplyLambda(p => {
            var centered = p - size / 2;
            var dp = dot(centered, normal_tensor);
            return dp.item<float>() > 0 ? 1 : 0;
        });
    }

    public Tensor GeneratePlane() {
        var normal = new Vector3(
            GD.Randf() - .5f,
            GD.Randf() - .5f,
            GD.Randf() - .5f
        ).Normalized();

        return GeneratePlane(normal);
    }

    private Tensor ApplyLambda(VoxelLambda lambda) {
        var data = zeros(new long[] { size, size, size });

        var t = zeros(3);

        for (var z = 0; z < size; z ++) {
            for (var y = 0; y < size; y ++) {
                for (var x = 0; x < size; x ++) {
                    t[0] = x;
                    t[1] = y;
                    t[2] = z;
                    data[z, y, x] = lambda(t);
                }
            }
        }

        return data;
    }

    public Tensor GenerateFromClass(long c) => c switch {
        0L => GenerateRandom(),
        1L => GeneratePlane(),
        2L => GenerateWorley(4),
        _ => throw new Exception("Invalid class token")
    };


    public (Tensor x, Tensor y) GenerateRandomBatch(int batchSize) {       
        var y = randint(0, num_classes, batchSize);

        var x_list = new System.Collections.Generic.List<Tensor>();

        for (var i = 0; i < batchSize; i ++) {
            x_list.Add(GenerateFromClass(y[i].item<long>()).unsqueeze(0));
        }

        return (cat(x_list, 0).cuda(), y.cuda());
    }

}
