using TorchSharp;

public static class GlobalDevice {
    public static readonly torch.Device device = torch.device(torch.cuda.is_available() ? "cuda" : "cpu");
}