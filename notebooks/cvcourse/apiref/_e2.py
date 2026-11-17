"""Reference entries for functions first used in E2 (bonus experiment: an FFT layer instead of a convolution).

Same format as `_day0.py`. Entries whose key already exists in an earlier file are ignored (the earliest
wording wins), so only functions no earlier notebook explains are here.
"""
from . import E

ENTRIES = [
    E("torch.fft.fft2", "torch.fft.fft2(x)",
      """The two-dimensional fast Fourier transform of the last two dimensions: rewrites a picture of H x W
      pixels as H x W waves. The result has the same shape and holds complex numbers: one per wave, saying how
      strong that wave is in the picture and how far it is shifted. Entry [0, 0] is the slowest "wave", a flat
      one: it equals the sum of all pixels.""",
      """
      >>> flat = torch.ones(2, 2)                               # a flat picture: nothing changes anywhere
      >>> torch.fft.fft2(flat)                                  # only the flat wave [0, 0] is present: 1+1+1+1 = 4
      tensor([[4.+0.j, 0.+0.j],
              [0.+0.j, 0.+0.j]])
      >>> stripes = torch.tensor([[1., -1.], [1., -1.]])        # changes from column to column
      >>> torch.fft.fft2(stripes)                               # only the wave "fast along the columns" is present
      tensor([[0.+0.j, 4.+0.j],
              [0.+0.j, 0.+0.j]])
      """,
      note="""The result is complex, so it cannot be drawn directly: draw its size with .abs(). The slow waves sit
      in the four corners, not in the middle; torch.fft.fftshift moves them to the middle.""",
      see=["torch.fft.ifft2", "torch.fft.rfft2", "torch.fft.fftshift"]),

    E("torch.fft.ifft2", "torch.fft.ifft2(spectrum)",
      """The inverse of torch.fft.fft2: adds the waves up again and gives back the picture. The result is still
      of a complex type; for a spectrum that came from a real picture its imaginary part is 0 (up to rounding),
      and .real takes the real part.""",
      """
      >>> x = torch.tensor([[1., 2.], [3., 4.]])
      >>> spectrum = torch.fft.fft2(x)
      >>> torch.fft.ifft2(spectrum).real                        # there and back: the picture again
      tensor([[1., 2.],
              [3., 4.]])
      """,
      see=["torch.fft.fft2", "torch.fft.irfft2"]),

    E("torch.fft.fftshift", "torch.fft.fftshift(x, dim=None)",
      """Rolls every dimension (or the dimensions in dim) by half its length, so that entry [0, 0] moves to the
      centre. A spectrum from torch.fft.fft2 has its slow waves in the corners; after fftshift they are in the
      middle, which is how spectra are usually drawn.""",
      """
      >>> torch.fft.fftshift(torch.arange(6.))                  # entry 0 moves to the middle
      tensor([3., 4., 5., 0., 1., 2.])
      >>> x = torch.zeros(4, 4)
      >>> x[0, 0] = 1.0
      >>> torch.fft.fftshift(x)
      tensor([[0., 0., 0., 0.],
              [0., 0., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 0.]])
      """,
      note="""Without dim it shifts every dimension, also the batch and channel dimensions: on a 4-dimensional
      tensor write dim=(2, 3).""",
      see=["torch.fft.ifftshift", "torch.roll"]),

    E("torch.fft.ifftshift", "torch.fft.ifftshift(x, dim=None)",
      """Undoes torch.fft.fftshift: moves the centre back to entry [0, 0]. Call it on a shifted spectrum before
      torch.fft.ifft2, which expects the slow waves in the corners.""",
      """
      >>> shifted = torch.fft.fftshift(torch.arange(6.))
      >>> shifted
      tensor([3., 4., 5., 0., 1., 2.])
      >>> torch.fft.ifftshift(shifted)
      tensor([0., 1., 2., 3., 4., 5.])
      """,
      see=["torch.fft.fftshift"]),

    E("torch.fft.rfft2", "torch.fft.rfft2(x)",
      """torch.fft.fft2 for pictures of real numbers, storing half of the result. The spectrum of a real picture
      is symmetric: the right half of its columns repeats the left half, so rfft2 keeps only W // 2 + 1 columns:
      (B, C, 28, 28) -> (B, C, 28, 15), complex. Nothing is lost: torch.fft.irfft2 rebuilds the picture.""",
      """
      >>> x = torch.randn(1, 1, 28, 28)
      >>> torch.fft.fft2(x).shape
      torch.Size([1, 1, 28, 28])
      >>> torch.fft.rfft2(x).shape                              # 28 // 2 + 1 = 15 columns
      torch.Size([1, 1, 28, 15])
      >>> torch.allclose(torch.fft.rfft2(x), torch.fft.fft2(x)[:, :, :, 0:15], atol=1e-4)     # the same numbers
      True
      """,
      see=["torch.fft.irfft2", "torch.fft.fft2"]),

    E("torch.fft.irfft2", "torch.fft.irfft2(spectrum, s=(H, W))",
      """The inverse of torch.fft.rfft2: from the half spectrum back to a picture of real numbers of H x W
      pixels: (B, C, 28, 15) complex -> (B, C, 28, 28) real.""",
      """
      >>> x = torch.randn(1, 1, 28, 28)
      >>> spectrum = torch.fft.rfft2(x)
      >>> back = torch.fft.irfft2(spectrum, s=(28, 28))
      >>> back.shape
      torch.Size([1, 1, 28, 28])
      >>> torch.allclose(back, x, atol=1e-5)                    # there and back: the picture again
      True
      """,
      note="""Always pass s=(H, W). From 15 columns alone PyTorch cannot tell whether the picture was 28 or 29
      pixels wide, and it guesses the even number.""",
      see=["torch.fft.rfft2"]),

    E("torch.complex", "torch.complex(real, imag)",
      """Builds a tensor of complex numbers from two real tensors of the same shape: real + imag * j. A complex
      number is a pair of ordinary numbers that is added and multiplied as one; a spectrum uses one per wave.""",
      """
      >>> real = torch.tensor([1., 2.])
      >>> imag = torch.tensor([3., 4.])
      >>> z = torch.complex(real, imag)
      >>> z
      tensor([1.+3.j, 2.+4.j])
      >>> z.real, z.imag                                        # the two parts again
      (tensor([1., 2.]), tensor([3., 4.]))
      >>> z.abs()                                               # the size: sqrt(1*1 + 3*3), sqrt(2*2 + 4*4)
      tensor([3.1623, 4.4721])
      """,
      see=["torch.fft.rfft2"]),

    E("F.conv2d", "F.conv2d(x, weight, bias=None, stride=1, padding=0)",
      """The convolution itself, as a function: slides the filters in weight over x. nn.Conv2d is this function
      plus a weight and a bias that it creates and trains; use F.conv2d when you have the filters yourself.
      x: (B, C_in, H, W). weight: (C_out, C_in, kH, kW). Result: (B, C_out, H', W').""",
      """
      >>> x = torch.arange(9.).reshape(1, 1, 3, 3)
      >>> x
      tensor([[[[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]]]])
      >>> weight = torch.ones(1, 1, 2, 2)                       # one filter of 2 x 2 that adds up its window
      >>> F.conv2d(x, weight)                                   # 0 + 1 + 3 + 4 = 8, ...
      tensor([[[[ 8., 12.],
                [20., 24.]]]])
      """,
      note="""Both arguments need all four dimensions: a single 3 x 3 filter must be reshaped to (1, 1, 3, 3).""",
      see=["nn.Conv2d", "F.pad"]),

    E("F.pad", "F.pad(x, (left, right, top, bottom), mode=\"constant\", value=0)",
      """Adds a border around the last two dimensions. mode="constant" fills it with value (zeros by default);
      mode="circular" fills it with the pixels from the opposite edge, as if the picture were a tile repeated
      in all directions.""",
      """
      >>> x = torch.arange(4.).reshape(1, 1, 2, 2)
      >>> x
      tensor([[[[0., 1.],
                [2., 3.]]]])
      >>> F.pad(x, (1, 1, 1, 1))                                # a border of zeros, 1 pixel wide
      tensor([[[[0., 0., 0., 0.],
                [0., 0., 1., 0.],
                [0., 2., 3., 0.],
                [0., 0., 0., 0.]]]])
      >>> F.pad(x, (1, 1, 1, 1), mode="circular")               # the border wraps around
      tensor([[[[3., 2., 3., 2.],
                [1., 0., 1., 0.],
                [3., 2., 3., 2.],
                [1., 0., 1., 0.]]]])
      """,
      note="""The order of the four numbers is left, right, top, bottom: the last dimension (width) comes first.""",
      see=["nn.Conv2d", "F.conv2d"]),
]
