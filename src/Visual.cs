using Godot;
using ScottPlot;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;

public partial class Visual : Node3D {

    [Export] private Sprite3D plotSprite;

    private Plot lossPlot;

    private readonly List<(long epoch, float loss, float accuracy)> lossList = new();

    private int lastListLength = 0;

    public override void _Ready() {
        lossPlot = new Plot(800, 600);
    }

    public void AddLossValue(long epoch, float loss, float accuracy) {
        lossList.Add((epoch, loss, accuracy));
    }

    public override void _Process(double delta) {

        if (lossList.Count > lastListLength) {
            lossPlot.Clear();

            var xs = new List<double>();
            var y1s = new List<double>();
            var y2s = new List<double>();

            foreach (var (epoch, loss, accuracy) in lossList) {
                xs.Add(epoch);
                y1s.Add(loss);
                y2s.Add(accuracy);
            }

            lossPlot.AddScatter(xs.ToArray(), y1s.ToArray());
            lossPlot.AddScatter(xs.ToArray(), y2s.ToArray(), color: System.Drawing.Color.Red);

            plotSprite.Texture = null;

            var render = lossPlot.Render();
            
            var image = new Image();

            using MemoryStream ms = new();
            render.Save(ms, ImageFormat.Png);
            ms.Position = 0;
            image.LoadPngFromBuffer(ms.ToArray());

            var texture = new ImageTexture();
            texture.SetImage(image);

            plotSprite.Texture = texture;

            lastListLength = lossList.Count;
        }

    }
}
