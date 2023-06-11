using Godot;
using System;
using System.Collections.Generic;

using static TorchSharp.torch;

delegate float VoxelLambda(Tensor position);

public partial class VoxelGenerator : Node3D {

    [Export] int voxel_size;
    [Export] int num_classes;
    [Export] int patch_size;

    public VoxelGenerator() : base() {}

    public int SetSize(int size) => voxel_size = size;

    public int SetNumClasses(int num_classes) => this.num_classes = num_classes;

    public int SetPatchSize(int patch_size) => this.patch_size = patch_size;

    public Tensor GenerateWorley(int n) {
        var center = randn(n, 3) * voxel_size;
        var radius = randn(n) * (voxel_size * 0.25f);

        return ApplyLambda(p => {
            var dp = center - p;
            var d = linalg.norm(dp, 2, dims: new long[] { 1 }) - radius;
            return d.max().item<float>() > 0 ? 1 : 0;
        });
    }

    public Tensor GenerateRandom() => randn(voxel_size, voxel_size, voxel_size).round();
    
    public Tensor GeneratePlane(Vector3 normal) {
        var normal_tensor = tensor(new float[] { normal.X, normal.Y, normal.Z });
        
        return ApplyLambda(p => {
            var centered = p - voxel_size / 2;
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
        var data = zeros(new long[] { voxel_size, voxel_size, voxel_size });

        var t = zeros(3);

        for (var z = 0; z < voxel_size; z ++) {
            for (var y = 0; y < voxel_size; y ++) {
                for (var x = 0; x < voxel_size; x ++) {
                    t[0] = x;
                    t[1] = y;
                    t[2] = z;
                    data[z, y, x] = lambda(t);
                }
            }
        }

        return data;
    }

    public Tensor ToPatches(Tensor voxelBatch) {
        var start_dim = 1;
        return voxelBatch
            .unfold(start_dim + 0, patch_size, patch_size)
            .unfold(start_dim + 1, patch_size, patch_size)
            .unfold(start_dim + 2, patch_size, patch_size)
            .flatten(start_dim, start_dim + 2)
            .flatten(-3, -1);
    }

    public Tensor GenerateFromClass(long c) => c switch {
        0L => GeneratePlane(Vector3.Right),
        1L => GeneratePlane(Vector3.Forward),
        2L => GeneratePlane(Vector3.Up),
        3L => GeneratePlane(Vector3.Left),
        _ => throw new Exception("Invalid class token")
    };

    public (Tensor x, Tensor y) GenerateRandomBatch(int batchSize, bool cuda = false) {       
        var y = randint(0, num_classes, batchSize);

        var x_list = new List<Tensor>();

        for (var i = 0; i < batchSize; i ++) {
            x_list.Add(GenerateFromClass(y[i].item<long>()).unsqueeze(0));
        }

        var x = ToPatches(cat(x_list, 0));

        return cuda ? (x.cuda(), y.cuda()) : (x, y);
    }

}
