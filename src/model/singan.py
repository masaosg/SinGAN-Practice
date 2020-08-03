from torch import nn

def init_weights(m):
    if type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, out_chan_init: int, out_chan_min: int):
        super().__init__()
        if out_chan_init < out_chan_min:
            raise ValueError(f'out_chan_init should be greater than or equal to out_chan_min {out_chan_min}')
        out_channels = [out_chan_init // pow(2, i) for i in range(4)]
        self.main = nn.Sequential(
            nn.Conv2d(3, out_channels[0], 3, 1, 0),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(out_channels[0], max(out_channels[1], out_chan_min), 3, 1, 0),
            nn.BatchNorm2d(max(out_channels[1], out_chan_min)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(max(out_channels[1], out_chan_min), max(out_channels[2], out_chan_min), 3, 1, 0),
            nn.BatchNorm2d(max(out_channels[2], out_chan_min)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(max(out_channels[2], out_chan_min), max(out_channels[3], out_chan_min), 3, 1, 0),
            nn.BatchNorm2d(max(out_channels[3], out_chan_min)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(max(out_channels[3], out_chan_min), 3, 3, 1, 0),
            nn.Tanh()
        )
    
    def forward(self, input1, input2):
        input1 = self.main(input1)
        diff2 = (input2.shape[2] - input1.shape[2]) // 2
        input2 = input2[:,:,diff2:(input2.shape[2] - diff2),diff2:(input2.shape[3] - diff2)]
        return input1 + input2

class Discriminator(nn.Module):
    def __init__(self, out_chan_init: int, out_chan_min: int):
        super().__init__()
        if out_chan_init < out_chan_min:
            raise ValueError(f'out_chan_init should be greater than or equal to out_chan_min {out_chan_min}')
        out_channels = [out_chan_init // pow(2, i) for i in range(4)]
        self.main = nn.Sequential(
            nn.Conv2d(3, out_channels[0], 3, 1, 0),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(out_channels[0], max(out_channels[1], out_chan_min), 3, 1, 0),
            nn.BatchNorm2d(max(out_channels[1], out_chan_min)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(max(out_channels[1], out_chan_min), max(out_channels[2], out_chan_min), 3, 1, 0),
            nn.BatchNorm2d(max(out_channels[2], out_chan_min)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(max(out_channels[2], out_chan_min), max(out_channels[3], out_chan_min), 3, 1, 0),
            nn.BatchNorm2d(max(out_channels[3], out_chan_min)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(max(out_channels[3], out_chan_min), 1, 3, 1, 0),
        )

    def forward(self, input1):
        return self.main(input1)
