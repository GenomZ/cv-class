from nb import md, code, api

NAME = "E2-fft-instead-of-convolution"

CELLS = [
    md(r"""
    # E2 · Bonus experiment: an FFT layer instead of a convolution

    This is a **bonus** notebook with one experiment. A convolution layer slides small filters over the picture.
    There is a second, very different way to compute the same thing: take the picture apart into **waves**,
    change the strength of every wave, and put the picture together again. The tool that takes a picture apart
    into waves is the **FFT** (fast Fourier transform).

    We build a layer that works in this way, put it into a network in the place of the convolution layers, and
    compare that **FFT-CNN** with the ordinary CNN next to it.

    *Hypothesis: a network that learns one number per wave, instead of 3 x 3 numbers per filter, classifies the
    digits as well as the CNN.*

    The plan:

    1. what the FFT does to a picture;
    2. the rule that connects the FFT with convolution, checked with numbers;
    3. the FFT layer;
    4. the comparison: an ordinary CNN against two FFT-CNNs;
    5. the same comparison repeated with several seeds, and with little data;
    6. a look at what the FFT layers learned;
    7. settings to try.

    **How to use this notebook.** As in P1 and E1 you do not write code: run the cells from top to bottom with
    **Shift+Enter**, read what comes out, and change the numbers in the cells marked **Try it**. Run P1 first:
    the words explained there (layer, filter, feature map, epoch, accuracy, ...) are used here without being
    explained again. The whole notebook runs in about 4 to 8 minutes on a laptop; every training cell says how
    long it takes.
    """),

    # ------------------------------------------------------------------ 0
    md(r"""
    ## 0 · Setup and data

    The same first cell as in E1, with two more imports: `math` (for a square root) and
    `torch.nn.functional`, short `F`, which holds the layers of PyTorch as plain functions. Then the same MNIST
    pictures as in P1: 10 000 for training, 2 000 for validation, in batches of 64.
    """),
    code("""
    import sys
    import math
    import pathlib

    folder = pathlib.Path.cwd()
    while not (folder / "cvcourse").is_dir() and folder != folder.parent:
        folder = folder.parent                      # go up until the folder that contains cvcourse
    sys.path.insert(0, str(folder))

    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from cvcourse import practical as P

    P.setup(seed=0)
    train_loader, val_loader, classes = P.get_mnist(n_train=10000, n_val=2000, batch_size=64)
    """),

    # ------------------------------------------------------------------ 1
    md(r"""
    ## 1 · A picture as a sum of waves

    ### 1.1 · The idea

    We are used to describing a picture pixel by pixel: 28 x 28 numbers, each saying how bright one place is.
    The same picture can be described in another way: as a **sum of waves**. A wave is a regular pattern of
    stripes that goes from dark to bright and back. A **slow** wave has wide stripes (it changes little from
    pixel to pixel); a **fast** wave has thin stripes (it changes a lot from pixel to pixel). The number of
    stripes that fit into the picture is the wave's **frequency**.

    A picture of 28 x 28 pixels can be written exactly as a sum of 28 x 28 such waves. For every wave we need
    to know two things: how **strong** it is in this picture, and how far it is **shifted** sideways. The list
    of these values for all waves is the **spectrum** of the picture. The **Fourier transform** computes the
    spectrum from the pixels; the **FFT** (fast Fourier transform) is the quick algorithm for it that every
    library uses. The **inverse** FFT goes back: from the spectrum to the pixels. Nothing is lost on the way
    there and back.

    A hand-worked example with tiny pictures of 2 x 2 pixels. There are 2 x 2 = 4 waves:

    ```
    flat             changes along      changes along      changes along
                     the columns        the rows           both
     1  1             1 -1               1  1               1 -1
     1  1             1 -1              -1 -1              -1  1
    ```

    The strength of a wave in a picture: multiply the picture by the wave, pixel by pixel, and add up.

    ```
    picture       flat wave:          1*5 + 1*3 + 1*5 + 1*3 = 16
     5  3         column wave:        1*5 - 1*3 + 1*5 - 1*3 =  4
     5  3         row wave:           1*5 + 1*3 - 1*5 - 1*3 =  0
                  both:               1*5 - 1*3 - 1*5 + 1*3 =  0
    ```

    The spectrum is `16, 4, 0, 0`. It says: the picture is bright overall (16), its left column is brighter
    than its right column (4), and nothing changes from the top row to the bottom row (0, 0). The same
    information as the four pixels, told differently.

    **Where the two values per wave are kept.** For bigger pictures the shift of a wave matters too, so the
    FFT returns one **complex number** per wave. A complex number is a pair of ordinary numbers, called its
    real and its imaginary part, written `3 + 4j`. Its **size** (`.abs()`: here `sqrt(3*3 + 4*4) = 5`) is the
    strength of the wave; the proportion between the two parts encodes the shift. That is all we need to know
    about complex numbers here: **a pair of numbers per wave**.

    ### 1.2 · The spectrum of a digit

    `torch.fft.fft2` computes the spectrum of a picture. Two details of the next cell:

    * The FFT puts the slowest waves into the corners of its result. `torch.fft.fftshift` moves them to the
      centre, which is easier to look at: after it the centre holds the slow waves and the border the fast ones.
    * A few waves are thousands of times stronger than the rest, so we draw `log(1 + strength)`: the logarithm
      squeezes large numbers together and makes the weak waves visible.
    """),
    api("torch.fft.fft2", "torch.fft.fftshift"),
    api("Tensor.abs", "torch.log", "Tensor.shape", brief=True),
    code("""
    image, label = val_loader.dataset[0]                 # one picture with its label: shape [1, 28, 28]
    picture = image[0]                                   # without the channel axis: shape [28, 28]

    spectrum = torch.fft.fft2(picture)                   # 28 x 28 complex numbers: one per wave
    spectrum = torch.fft.fftshift(spectrum)              # move the slow waves to the centre
    strength = spectrum.abs()                            # the size of every complex number

    print("shape of the picture: ", picture.shape)
    print("shape of the spectrum:", spectrum.shape)
    print("one entry of the spectrum:", spectrum[14, 15])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), layout="constrained")
    axes[0].imshow(picture, cmap="gray")
    axes[0].set_title("the picture: 28 x 28 pixels")
    axes[0].axis("off")
    axes[1].imshow(torch.log(1 + strength), cmap="magma")
    axes[1].set_title("its spectrum: the strength of 28 x 28 waves")
    axes[1].axis("off")
    plt.show()
    """),
    md(r"""
    **What to look at.** The spectrum is bright in the centre and dark towards the border: the digit is made
    mostly of **slow** waves. Nothing in the spectrum looks like a digit: every wave covers the whole picture,
    so every number of the spectrum says something about the whole picture and nothing about one place.

    ### 1.3 · What slow and fast waves hold

    To see what the waves are for, we throw some away. The next cell keeps only the waves in a small square
    around the centre of the spectrum (the slowest ones), sets all others to 0 and goes back to pixels with the
    inverse FFT:

    1. `mask` is a 28 x 28 grid of zeros with ones in the square we keep;
    2. `spectrum * mask` sets every wave outside the square to 0;
    3. `torch.fft.ifftshift` undoes the `fftshift` (slow waves back into the corners);
    4. `torch.fft.ifft2` adds the remaining waves up to a picture. Its result is still of a complex type; `.real`
       takes the real part (the imaginary part is 0 here).

    With `reach = 14` the square covers the whole spectrum and the picture comes back unchanged.
    """),
    api("torch.fft.ifft2", "torch.fft.ifftshift"),
    api("torch.zeros", brief=True),
    code("""
    reaches = [1, 3, 6, 14]                              # how far from the centre we keep the waves

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), layout="constrained")
    for position in range(4):
        reach = reaches[position]
        first = max(14 - reach, 0)                       # first row and column of the square we keep
        last = min(14 + reach + 1, 28)                   # one past its last row and column
        mask = torch.zeros(28, 28)
        mask[first:last, first:last] = 1.0               # 1 = keep this wave, 0 = throw it away
        kept = spectrum * mask
        kept = torch.fft.ifftshift(kept)                 # slow waves back into the corners
        rebuilt = torch.fft.ifft2(kept).real             # add the kept waves up: a picture again
        n_waves = (last - first) * (last - first)
        axes[position].imshow(rebuilt, cmap="gray")
        if n_waves == 784:
            axes[position].set_title("all 784 waves")
        else:
            axes[position].set_title(f"the {n_waves} slowest waves of 784")
        axes[position].axis("off")
    plt.show()
    """),
    md(r"""
    **What to look at.** 9 waves give a blurred blob in the right place. 49 waves (6% of all) already show
    which digit it is. 169 waves give a soft but complete digit. The remaining fast waves only sharpen the
    edges of the strokes. **Slow waves hold the rough shape, fast waves hold the fine detail.**
    """),

    # ------------------------------------------------------------------ 2
    md(r"""
    ## 2 · Convolution is a multiplication of spectra

    ### 2.1 · The rule

    What does a convolution do to the waves of a picture? Slide any filter over a wave and you get **the same
    wave** again, only stronger or weaker, and shifted. A filter cannot turn one wave into another. So a filter
    is fully described by a list: for every wave, by how much it strengthens and shifts it. That list is the
    **spectrum of the filter**. This gives the rule called the **convolution theorem**:

    > Sliding a filter over a picture gives the same result as multiplying the spectrum of the picture by the
    > spectrum of the filter, wave by wave.

    So there are two ways to the same feature map:

    ```
    way 1 (slide):      picture ------------------ slide the 3 x 3 filter ------------------> feature map

    way 2 (multiply):   picture --FFT--> spectrum --multiply by the filter's spectrum--> --inverse FFT--> feature map
    ```

    Three details are needed to make the two ways agree **exactly**:

    1. **The filter must be as large as the picture** before its FFT is taken, because we multiply two spectra
       of the same shape. We lay the 3 x 3 filter on an empty (all zeros) picture of 28 x 28.
    2. **The centre of the filter must lie on pixel (0, 0)**, the top left corner. Its left column and upper row
       then stick out of the picture and come back in on the opposite side. `torch.roll` moves a picture
       sideways in exactly this wrap-around way.
    3. **The filter must be flipped** (turned by 180 degrees). The convolution of the theorem, as mathematicians
       define it, flips the filter before sliding it; `nn.Conv2d` does not. Flipping it ourselves first cancels
       that.

    One more thing follows from the waves: a wave has no beginning and no end, it goes around the picture.
    For the FFT the picture is a **tile** that repeats in all directions: right of the last column comes the
    first column again. A filter at the right edge therefore sees pixels of the left edge. An ordinary
    convolution with `padding=1` puts a border of zeros around the picture instead. To compare the two ways
    fairly, way 1 must use the wrap-around border too: `F.pad(x, (1, 1, 1, 1), mode="circular")`. For MNIST
    this changes almost nothing, because all four edges of a digit picture are background.

    ### 2.2 · From a filter to its spectrum

    `kernel_to_spectrum` does the three steps and then the FFT. It takes filters in the shape in which a
    convolution layer keeps them, `[out_channels, in_channels, 3, 3]`, and returns one spectrum per filter.

    A hand-worked example of steps 1 to 3 for the Sobel x filter of E1 and a picture of 5 x 5:

    ```
    the filter        1. flipped        2. on an empty 5 x 5 picture     3. rolled 1 up and 1 left:
                                           (top left corner)                the centre (0) is at pixel (0, 0)
    -1  0  1           1  0 -1           1  0 -1  0  0                    0 -2  0  0  2
    -2  0  2           2  0 -2           2  0 -2  0  0                    0 -1  0  0  1
    -1  0  1           1  0 -1           1  0 -1  0  0                    0  0  0  0  0
                                         0  0  0  0  0                    0  0  0  0  0
                                         0  0  0  0  0                    0 -1  0  0  1
    ```

    The last step uses `torch.fft.rfft2` instead of `torch.fft.fft2`. It computes the same spectrum and stores
    half of it: the spectrum of a picture of real numbers is symmetric (its right half repeats its left half),
    so 15 of the 28 columns are enough. Its partner for the way back is `torch.fft.irfft2`.

    | step | shape for one Sobel filter and a 28 x 28 picture |
    |---|---|
    | `weight` | `[1, 1, 3, 3]` |
    | `flipped` | `[1, 1, 3, 3]` |
    | `big` | `[1, 1, 28, 28]` |
    | the result of `torch.fft.rfft2(big)` | `[1, 1, 28, 15]`, complex |
    """),
    api("torch.fft.rfft2"),
    api("Tensor.flip", "torch.roll", brief=True),
    code("""
    def kernel_to_spectrum(weight, height, width):
        n_out = weight.shape[0]                              # how many filters
        n_in = weight.shape[1]                               # input channels of every filter
        k = weight.shape[2]                                  # the kernel size: 3
        # 1. flip every filter (turn it by 180 degrees)
        flipped = weight.flip(dims=(2, 3))
        # 2. lay it on an empty picture of the full size, in the top left corner
        big = torch.zeros(n_out, n_in, height, width)
        big[:, :, 0:k, 0:k] = flipped
        # 3. move it up and left until its centre lies on pixel (0, 0); what sticks out wraps around
        half = k // 2                                        # 1 for a 3 x 3 filter
        big = torch.roll(big, shifts=(-half, -half), dims=(2, 3))
        # 4. the spectrum of the big filter: one complex number per wave
        return torch.fft.rfft2(big)
    """),
    md(r"""
    ### 2.3 · Check: both ways give the same feature map

    We filter the digit with Sobel x in both ways and compare.

    * **Way 1:** `F.conv2d(x, weight)` is the convolution as a plain function: it slides the filters in `weight`
      over `x`. (`nn.Conv2d` is the same function together with a weight that it creates and trains itself.)
      Both arguments need four dimensions, so the picture becomes `[1, 1, 28, 28]` (a batch with one picture)
      and the filter `[1, 1, 3, 3]`.
    * **Way 2:** FFT of the picture, multiply by the spectrum of the filter, inverse FFT.

    The last line prints the largest difference between the two feature maps over all 784 pixels.
    """),
    api("torch.fft.irfft2", "F.conv2d", "F.pad"),
    api("torch.tensor", "Tensor.reshape", "Tensor.unsqueeze", "Tensor.max", "Tensor.item", brief=True),
    code("""
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]])
    sobel_weight = sobel_x.reshape(1, 1, 3, 3)           # 1 filter with 1 input channel
    x = image.unsqueeze(0)                               # a batch with one picture: [1, 1, 28, 28]

    # way 1: slide the filter over the picture (with a wrap-around border)
    padded = F.pad(x, (1, 1, 1, 1), mode="circular")     # [1, 1, 30, 30]
    map_slide = F.conv2d(padded, sobel_weight)           # [1, 1, 28, 28]

    # way 2: multiply the two spectra
    picture_spectrum = torch.fft.rfft2(x)                             # [1, 1, 28, 15], complex
    filter_spectrum = kernel_to_spectrum(sobel_weight, 28, 28)        # [1, 1, 28, 15], complex
    product = picture_spectrum * filter_spectrum                      # wave by wave
    map_fft = torch.fft.irfft2(product, s=(28, 28))                   # back to pixels: [1, 1, 28, 28]

    difference = (map_slide - map_fft).abs().max().item()
    print("largest value in the feature map:      ", map_slide.abs().max().item())
    print("largest difference between the two ways:", difference)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
    axes[0].imshow(picture, cmap="gray")
    axes[0].set_title("the picture")
    axes[1].imshow(map_slide[0, 0], cmap="bwr", vmin=-12, vmax=12)
    axes[1].set_title("way 1: Sobel x by sliding")
    axes[2].imshow(map_fft[0, 0], cmap="bwr", vmin=-12, vmax=12)
    axes[2].set_title("way 2: Sobel x by FFT")
    for ax in axes:
        ax.axis("off")
    plt.show()
    """),
    md(r"""
    **What to look at.** The two feature maps are the same picture, and the largest difference is a few
    millionths: that is the rounding error of the computer's arithmetic, not a difference between the methods.
    The convolution theorem holds: **an FFT, a multiplication and an inverse FFT are a convolution.**
    """),

    # ------------------------------------------------------------------ 3
    md(r"""
    ## 3 · The FFT layer

    ### 3.1 · What the layer learns

    In section 2 the filter's spectrum was computed from a 3 x 3 filter. Now the step that makes it a new kind
    of layer: **we do not keep a 3 x 3 filter at all. The layer keeps the spectrum itself as its weights and
    training changes it directly**: one complex number per wave.

    What this means:

    * A spectrum has 28 x 15 complex numbers = 28 x 15 x 2 = **840** ordinary numbers. A 3 x 3 filter has **9**.
      One "filter" of the FFT layer has 93 times as many weights.
    * A spectrum of 28 x 15 free numbers is the spectrum of a filter of **28 x 28** pixels. The FFT layer is
      therefore a convolution whose filter is **as large as the whole picture** (with wrap-around). One output
      pixel of the first layer can depend on every pixel of the picture; in a CNN it depends on 3 x 3 pixels.
    * A 3 x 3 filter is one special case among all these spectra. Anything a 3 x 3 convolution can compute,
      the FFT layer can compute too; the reverse is not true.

    Like a convolution layer, the FFT layer has several input and output channels and one filter spectrum for
    every pair (output channel, input channel). An output channel is the sum over the input channels, each
    multiplied by its own filter spectrum, plus a bias.

    **The weights.** PyTorch's optimisers are most at home with ordinary (real) numbers, so the layer keeps the
    two parts of the complex spectrum as two ordinary tensors, `weight_real` and `weight_imag`, each of shape
    `[out_channels, in_channels, height, width // 2 + 1]`. `torch.complex(weight_real, weight_imag)` joins
    them into complex numbers when they are needed. `nn.Parameter(tensor)` marks a tensor as a weight of the
    layer: something that training has to change.

    **Two ways to start.** The argument `start` says what the spectra are before training:

    * `start="spectrum"`: **random numbers for every wave.** The matching 28 x 28 filter is random noise all
      over the picture. `torch.randn` draws numbers of typical size 1; we divide them by
      `sqrt(in_channels)`, because an output channel adds up `in_channels` products and would otherwise grow
      with the number of input channels.
    * `start="3x3"`: **the spectrum of an ordinary random 3 x 3 filter.** We create an `nn.Conv2d`, borrow its
      random 3 x 3 filters and its biases, and convert the filters with `kernel_to_spectrum`. Before training
      the layer then computes exactly what that convolution layer computes. Training is free to leave the
      3 x 3 shape afterwards.

    **Shapes in `forward`** for the first layer (1 input channel, 8 output channels) and a batch of 64 pictures:

    | step | shape | what it is |
    |---|---|---|
    | `x` | `[64, 1, 28, 28]` | 64 pictures, 1 channel |
    | `spectrum = torch.fft.rfft2(x)` | `[64, 1, 28, 15]`, complex | the spectrum of every channel of every picture |
    | `spectrum.unsqueeze(1)` | `[64, 1, 1, 28, 15]` | a new axis for the output channels |
    | `weight` | `[8, 1, 28, 15]`, complex | one filter spectrum per (output, input) pair |
    | `product = spectrum * weight` | `[64, 8, 1, 28, 15]` | every picture with every filter, wave by wave |
    | `summed = product.sum(dim=2)` | `[64, 8, 28, 15]` | added up over the input channels |
    | `maps = torch.fft.irfft2(summed, s=(28, 28))` | `[64, 8, 28, 28]` | back to pixels: 8 feature maps per picture |
    | `maps + bias` | `[64, 8, 28, 28]` | the bias, reshaped to `[1, 8, 1, 1]`, is added to every pixel of its map |

    The multiplication in row 5 works by **broadcasting** (D0): `[64, 1, 1, 28, 15]` times `[8, 1, 28, 15]`.
    PyTorch lines the shapes up from the right and stretches every axis of length 1.
    """),
    api("torch.complex"),
    api("nn.Module", "nn.Parameter", "nn.Conv2d", "torch.randn", "Tensor.detach", "Tensor.clone", "Tensor.sum", brief=True),
    code("""
    class FFTConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, height, width, start):
            super().__init__()
            n_columns = width // 2 + 1                       # rfft2 keeps this many columns of the spectrum
            if start == "spectrum":
                # random numbers for every wave
                scale = 1.0 / math.sqrt(in_channels)
                real = torch.randn(out_channels, in_channels, height, n_columns) * scale
                imag = torch.randn(out_channels, in_channels, height, n_columns) * scale
                bias = torch.zeros(out_channels)
            else:
                # start="3x3": the spectrum of the random 3 x 3 filters of an ordinary convolution layer
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
                filter_spectrum = kernel_to_spectrum(conv.weight.detach(), height, width)
                real = filter_spectrum.real.clone()
                imag = filter_spectrum.imag.clone()
                bias = conv.bias.detach().clone()
            self.weight_real = nn.Parameter(real)            # the real parts: [out, in, height, n_columns]
            self.weight_imag = nn.Parameter(imag)            # the imaginary parts: the same shape
            self.bias = nn.Parameter(bias)                   # one bias per output channel

        def forward(self, x):
            height = x.shape[2]
            width = x.shape[3]
            # 1. pixels -> waves
            spectrum = torch.fft.rfft2(x)                                    # [B, in, H, W // 2 + 1]
            # 2. multiply every input channel by its filter spectrum, wave by wave
            weight = torch.complex(self.weight_real, self.weight_imag)       # [out, in, H, W // 2 + 1]
            product = spectrum.unsqueeze(1) * weight                         # [B, out, in, H, W // 2 + 1]
            # 3. add up over the input channels
            summed = product.sum(dim=2)                                      # [B, out, H, W // 2 + 1]
            # 4. waves -> pixels
            maps = torch.fft.irfft2(summed, s=(height, width))               # [B, out, H, W]
            # 5. add the bias of every output channel
            return maps + self.bias.reshape(1, -1, 1, 1)
    """),
    md(r"""
    Note what is **not** in this layer: no kernel size, no padding, no stride. But it needs the `height` and
    `width` of its input when it is created, because it has one weight per wave and the number of waves depends
    on the size of the picture. A convolution layer works on pictures of any size; this layer works on one size.

    ### 3.2 · Two networks, layer for layer

    The CNN is the one of E1 (experiment 2): two convolutions with 8 and 32 filters, each followed by ReLU and
    max pooling, then two linear layers. The only change is `padding_mode="circular"`: the wrap-around border
    of section 2.1, so that the CNN and the FFT-CNN treat the edges in the same way.

    The FFT-CNN is the same network with every `nn.Conv2d` replaced by an `FFTConv2d`. ReLU, pooling and the
    linear layers stay as they are and work on pixels: every FFT layer goes to the waves and comes back.

    | CNN | FFT-CNN | output shape |
    |---|---|---|
    | `nn.Conv2d(1, 8, kernel_size=3, padding=1, padding_mode="circular")` | `FFTConv2d(1, 8, 28, 28, start)` | 8 x 28 x 28 |
    | `nn.ReLU()`, `nn.MaxPool2d(2)` | the same | 8 x 14 x 14 |
    | `nn.Conv2d(8, 32, kernel_size=3, padding=1, padding_mode="circular")` | `FFTConv2d(8, 32, 14, 14, start)` | 32 x 14 x 14 |
    | `nn.ReLU()`, `nn.MaxPool2d(2)` | the same | 32 x 7 x 7 |
    | `nn.Flatten()` | the same | 1568 |
    | `nn.Linear(1568, 64)`, `nn.ReLU()` | the same | 64 |
    | `nn.Linear(64, 10)` | the same | 10 |
    """),
    api("nn.Sequential", "nn.ReLU", "nn.MaxPool2d", "nn.Flatten", "nn.Linear", brief=True),
    code("""
    def build_cnn():
        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, padding_mode="circular"),      # 8 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                                                         # 8 x 14 x 14
            nn.Conv2d(8, 32, kernel_size=3, padding=1, padding_mode="circular"),     # 32 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),                                                         # 32 x 7 x 7
            nn.Flatten(),                                                            # 1568 numbers
            nn.Linear(1568, 64),
            nn.ReLU(),
            nn.Linear(64, 10),                                                       # 10 scores
        )


    def build_fft_cnn(start):
        return nn.Sequential(
            FFTConv2d(1, 8, 28, 28, start),                 # 8 x 28 x 28
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 8 x 14 x 14
            FFTConv2d(8, 32, 14, 14, start),                # 32 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 32 x 7 x 7
            nn.Flatten(),                                   # 1568 numbers
            nn.Linear(1568, 64),
            nn.ReLU(),
            nn.Linear(64, 10),                              # 10 scores
        )
    """),
    md(r"""
    Three networks. Each is built right after `torch.manual_seed(0)`, so each draws the same random numbers in
    the same order.

    For the FFT-CNN with `start="3x3"` this has a useful consequence. Its FFT layers create an `nn.Conv2d`
    inside and borrow its filters, so they draw exactly the random filters that the CNN draws, and the linear
    layers after them draw the same weights as well. The two networks are **twins**: before training they hold
    the same filters, one as 3 x 3 grids, the other as spectra. By the convolution theorem they must then give
    the same scores for every picture. The last lines check that on the first 64 validation pictures.
    """),
    api("torch.manual_seed", "torch.no_grad", "torch.stack", brief=True),
    code("""
    torch.manual_seed(0)
    cnn = build_cnn()

    torch.manual_seed(0)
    fft_cnn = build_fft_cnn(start="spectrum")            # random numbers for every wave

    torch.manual_seed(0)
    fft_cnn_3x3 = build_fft_cnn(start="3x3")             # the CNN's own 3 x 3 filters, kept as spectra

    pictures = []
    for number in range(64):
        one_image, one_label = val_loader.dataset[number]
        pictures.append(one_image)
    batch = torch.stack(pictures)                        # [64, 1, 28, 28]

    with torch.no_grad():                                # only computing scores, not training
        scores_cnn = cnn(batch)
        scores_fft = fft_cnn(batch)
        scores_3x3 = fft_cnn_3x3(batch)

    print("CNN against FFT-CNN, start 3x3:      largest difference of the scores:", (scores_cnn - scores_3x3).abs().max().item())
    print("CNN against FFT-CNN, start spectrum: largest difference of the scores:", (scores_cnn - scores_fft).abs().max().item())
    """),
    md(r"""
    The twin gives the same scores as the CNN up to rounding error: before training it **is** the CNN, computed
    through the FFT. The FFT-CNN with the random spectrum is a different network from the start.

    Now the diagram and the table of layers of the FFT-CNN.
    """),
    code("""
    P.show_model(fft_cnn, (1, 28, 28), title="FFT-CNN: an FFT layer in the place of every convolution")
    """),
    code("""
    P.summary(fft_cnn, (1, 28, 28))
    """),
    md(r"""
    **Counting parameters.** The first FFT layer has 8 x 1 filter spectra of 28 x 15 x 2 = 840 numbers and 8
    biases: 6 728. The second has 32 x 8 spectra of 14 x 8 x 2 = 224 numbers and 32 biases: 57 376. The
    convolutions they replace have 8 x 9 + 8 = 80 and 32 x 8 x 9 + 32 = 2 336.

    | | CNN | FFT-CNN |
    |---|---|---|
    | first layer | 80 | 6 728 |
    | second layer | 2 336 | 57 376 |
    | the two linear layers | 101 066 | 101 066 |
    | **together** | **103 482** | **165 170** |

    (The column "mult-adds" of the table shows 0 for the FFT layers: the counting tool knows convolutions and
    linear layers and does not know how to count the work of an FFT. Use the measured times below instead.)

    `P.compare` measures the cost of the two networks on this computer.
    """),
    code("""
    P.compare({"CNN": cnn, "FFT-CNN": fft_cnn}, (1, 28, 28))
    """),
    md(r"""
    **What to look at.** The FFT-CNN has 60% more parameters, and on the computer this text was written on a
    training step took it about 1.6 times as long as the CNN (9 ms against 6 ms for 32 pictures; your times
    will differ). For a 3 x 3 filter, sliding is cheap: 9 multiplications per pixel. The FFT layer transforms
    every channel there and back and multiplies every wave of every (output, input) pair, whatever the size of
    the filter. That is the price of a filter as large as the picture; it would be the same price for a 3 x 3,
    a 9 x 9 or a 27 x 27 filter, while the cost of sliding grows with the area of the filter.
    """),

    # ------------------------------------------------------------------ 4
    md(r"""
    ## 4 · The comparison

    All three networks are trained in the same way: the same pictures in the same order, 5 epochs, learning
    rate 0.001.

    **Expected time: about 1 minute** for the three networks together (up to 2 minutes on a slow laptop).
    """),
    code("""
    history_cnn = P.train(cnn, train_loader, val_loader, epochs=5, lr=0.001, name="CNN")
    history_fft = P.train(fft_cnn, train_loader, val_loader, epochs=5, lr=0.001, name="FFT-CNN, start spectrum")
    history_3x3 = P.train(fft_cnn_3x3, train_loader, val_loader, epochs=5, lr=0.001, name="FFT-CNN, start 3x3")
    """),
    code("""
    P.plot_history({"CNN": history_cnn, "FFT-CNN, start spectrum": history_fft, "FFT-CNN, start 3x3": history_3x3})
    """),
    md(r"""
    **What we saw** (your numbers may differ a little):

    | network | parameters | training accuracy after 5 epochs | best validation accuracy |
    |---|---|---|---|
    | CNN | 103 482 | 97.8% | 97.0% |
    | FFT-CNN, start spectrum | 165 170 | 98.2% | 96.0% |
    | FFT-CNN, start 3x3 | 165 170 | 98.3% | 97.2% |

    * **All three learn the digits.** A layer that never slides a filter, and only multiplies waves, can take
      the place of a convolution layer in a network that trains normally.
    * **The FFT-CNN with the random spectrum is one point behind**: 96.0% against 97.0%, that is 20 more wrong
      pictures among the 2 000. On the training pictures it is *ahead* of the CNN (98.2% against 97.8%). Better
      on the pictures it has seen, worse on new ones: the first sign of **overfitting**, learning the training
      pictures by heart.
    * **The twin with the 3 x 3 start ends 0.2 points above the CNN.** That is 4 pictures of 2 000: too little
      to believe from one run.

    ## 5 · Several seeds, and little data

    As in E1: a difference between two single runs may be chance. `repeat_runs` (the function of E1, section
    1.4, with the time added) trains one network per seed and prints the best validation accuracy of each run,
    their average, the spread between the best and the worst run, and the average training time.

    We also repeat everything with only **1 000 training pictures**. The question: the FFT-CNN has more
    freedom than the CNN (840 weights per filter instead of 9). Does it need more examples to use that freedom
    well, so that it falls further behind when the data is scarce?

    **Expected time: about 3 minutes** (up to 5 minutes on a slow laptop).
    """),
    code("""
    def repeat_runs(name, build_function, train_loader, val_loader, seeds, epochs):
        accuracies = []
        seconds = 0.0
        text = ""
        for seed in seeds:
            torch.manual_seed(seed)                      # other starting weights for every seed
            model = build_function()
            history = P.train(model, train_loader, val_loader, epochs=epochs, lr=0.001, seed=seed, verbose=False)
            accuracies.append(history["best_val_acc"])
            seconds = seconds + history["total_seconds"]
            text = text + f"{history['best_val_acc']:6.2f}"
        average = sum(accuracies) / len(accuracies)
        spread = max(accuracies) - min(accuracies)       # distance between the best and the worst run
        average_seconds = seconds / len(seeds)
        print(f"{name:<26} runs:{text}   average {average:.2f}%   spread {spread:.2f}   {average_seconds:.0f} s per run")


    def build_fft_spectrum():
        return build_fft_cnn(start="spectrum")

    def build_fft_3x3():
        return build_fft_cnn(start="3x3")


    seeds = [0, 1, 2]
    print("10 000 training pictures, 5 epochs")
    repeat_runs("CNN", build_cnn, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("FFT-CNN, start spectrum", build_fft_spectrum, train_loader, val_loader, seeds, epochs=5)
    repeat_runs("FFT-CNN, start 3x3", build_fft_3x3, train_loader, val_loader, seeds, epochs=5)

    small_train, small_val, classes = P.get_mnist(n_train=1000, n_val=2000, batch_size=64, verbose=False)
    print()
    print("1 000 training pictures, 5 epochs")
    repeat_runs("CNN", build_cnn, small_train, small_val, seeds, epochs=5)
    repeat_runs("FFT-CNN, start spectrum", build_fft_spectrum, small_train, small_val, seeds, epochs=5)
    repeat_runs("FFT-CNN, start 3x3", build_fft_3x3, small_train, small_val, seeds, epochs=5)
    """),
    md(r"""
    **What we saw** (best validation accuracy of three runs; your numbers may differ a little):

    | network | 10 000 pictures: average | spread | 1 000 pictures: average | spread |
    |---|---|---|---|---|
    | CNN | 96.98% | 0.50 | 88.73% | 1.80 |
    | FFT-CNN, start spectrum | 96.25% | 0.60 | 88.18% | 3.20 |
    | FFT-CNN, start 3x3 | 97.35% | 0.30 | 88.95% | 1.00 |

    **With 10 000 pictures:**

    * The FFT-CNN with the random spectrum is 0.7 points below the CNN, and that is more than the spread: its
      best run (96.55%) is below the CNN's worst run (96.75%). This difference is real.
    * The FFT-CNN with the 3 x 3 start is 0.4 points above the CNN. Its worst run (97.20%) is level with the
      CNN's best (97.25%). It is at least as good as the CNN, probably a little better; three seeds are not
      enough to be sure of 0.4 points.

    **With 1 000 pictures** all three lose about 8 points, and the differences between them (0.2 to 0.8 points)
    are smaller than the spread between seeds (1 to 3 points). These runs do **not** show that the FFT-CNN
    suffers more from little data. The one thing that stands out is the spread of the random-spectrum start:
    its three runs lie 3.2 points apart, so how well it does depends strongly on the random numbers it
    happened to start from.

    **The hypothesis** said that the FFT-CNN classifies as well as the CNN. The answer has two parts: yes, if it
    starts from the spectra of small filters; not quite, if it starts from random spectra. The layer is the
    same in both cases: what differs is only the **starting point**. Section 6 looks at why that matters so
    much here.

    (The times per run depend on what else the computer is doing; for the cost of the two kinds of layer trust
    the table of section 3.2, which measures them one right after the other.)
    """),

    # ------------------------------------------------------------------ 6
    md(r"""
    ## 6 · What did the FFT layers learn?

    The weights of an FFT layer are spectra, and section 1.2 showed that a spectrum is hard to read. But every
    spectrum belongs to a filter in pixels, and the inverse FFT gives it to us: the **28 x 28 filter that the
    layer would slide over the picture** if it were a convolution layer.

    `show_fft_filters` does that for the 8 filters of a first layer:

    1. join `weight_real` and `weight_imag` into complex spectra (`.detach()` hands us the numbers without the
       bookkeeping that training attaches to weights);
    2. inverse FFT: 8 filters of 28 x 28 pixels;
    3. `torch.fft.fftshift` over the two picture axes moves the centre of every filter from pixel (0, 0) to the
       middle of the picture (pixel (14, 14)), where it is easier to see;
    4. measure how much of each filter lies inside the central 3 x 3 pixels. We compare the sum of the squared
       weights inside the 3 x 3 square with the sum of all squared weights of the filters (squares, so that
       positive and negative weights do not cancel). 100% means: these are 3 x 3 filters.

    Red is a positive weight, blue a negative one, white is 0.
    """),
    code("""
    def show_fft_filters(layer, title):
        # 1. the filter spectra of the layer
        weight = torch.complex(layer.weight_real, layer.weight_imag).detach()     # [8, 1, 28, 15]
        # 2. back to pixels: the filters this layer would slide over the picture
        filters = torch.fft.irfft2(weight, s=(28, 28))                            # [8, 1, 28, 28]
        # 3. the centre of every filter from pixel (0, 0) to the middle of the picture
        filters = torch.fft.fftshift(filters, dim=(2, 3))
        # 4. the share of the squared weights that lies inside the central 3 x 3 pixels
        centre = filters[:, :, 13:16, 13:16]
        inside = (centre * centre).sum().item()
        everything = (filters * filters).sum().item()
        share = 100.0 * inside / everything

        fig, axes = plt.subplots(1, 8, figsize=(13, 2.2), layout="constrained")
        for number in range(8):
            one_filter = filters[number, 0]
            limit = one_filter.abs().max().item()
            axes[number].imshow(one_filter, cmap="bwr", vmin=-limit, vmax=limit)
            axes[number].axis("off")
        fig.suptitle(f"{title}: {share:.0f}% inside the central 3 x 3 pixels")
        plt.show()


    torch.manual_seed(0)
    untrained_3x3 = build_fft_cnn(start="3x3")           # the twin again, as it was before training
    torch.manual_seed(0)
    untrained_spectrum = build_fft_cnn(start="spectrum")

    show_fft_filters(untrained_3x3[0], "start 3x3, before training")
    show_fft_filters(fft_cnn_3x3[0], "start 3x3, after training")
    show_fft_filters(untrained_spectrum[0], "start spectrum, before training")
    show_fft_filters(fft_cnn[0], "start spectrum, after training")
    """),
    md(r"""
    **What we saw.**

    * **Start 3x3, before training:** every filter is a 3 x 3 square of weights in the middle of an empty
      28 x 28 picture: 100% inside. This is the CNN's filter, seen through the FFT layer.
    * **Start 3x3, after training:** still 98% inside. Training was free to put weights anywhere in the 28 x 28
      picture, and it used that freedom very little: a faint haze around the centre is all that was added. The
      network stayed close to a 3 x 3 CNN, and that is why it performs like one.
    * **Start spectrum, before training:** noise over the whole picture, 1% inside the centre. Such a filter
      mixes pixels from everywhere: one output pixel is a random blend of the whole digit.
    * **Start spectrum, after training:** the same noise. You have to look closely to find a pixel that
      changed. In 5 epochs the weights moved only a little away from where they started (one training step
      changes a weight by about 0.001, and these weights have size 1). The filters stayed random blends, and
      the layers after them learned to read the digit out of those blends. That works, to 96%, but less well
      than filters that look at neighbouring pixels.

    **Why the small filter is the better start.** In a picture the pixels that belong together are next to
    each other: a stroke, a corner, an edge are **local**. A 3 x 3 convolution has this knowledge built in: it
    *cannot* look anywhere else. The FFT layer has no such limit. It can represent local filters, but nothing
    in it says that local filters are a good idea, and training did not discover that by itself from 10 000
    digits. The limit of the convolution is not a weakness; it is knowledge about pictures given to the
    network for free.
    """),

    # ------------------------------------------------------------------ 7
    md(r"""
    ## 7 · Try it

    Change the settings, run this cell and the next one. The run cell trains a CNN and an FFT-CNN on the same
    data and draws both curves. Each experiment in the list is a question: predict the answer first.

    1. **Very little data**: `n_train = 300`. Which network suffers more?
    2. **A lot of data**: `n_train = 60000` (takes a few minutes). Does the FFT-CNN catch up?
    3. **One epoch only**: `epochs = 1`. Which start is ahead at the very beginning?
    4. **Longer training**: `epochs = 15`. Watch the gap between the training and the validation curve of each
       network: a network that learns the training pictures by heart (**overfitting**) keeps improving on
       them while its validation accuracy stands still.
    5. **A larger learning rate**: `learning_rate = 0.01`. Which network copes with it?
    6. **Harder pictures**: `dataset = "fashion"` (Fashion-MNIST: pictures of clothes, same size, 10 classes).
    """),
    code("""
    # ---------------- Try it: settings of the FFT-CNN ----------------
    start = "spectrum"            # "spectrum": random numbers for every wave; "3x3": the spectrum of random 3 x 3 filters
    # ---------------- run settings ----------------
    dataset = "mnist"             # "mnist" (digits) or "fashion" (clothes)
    n_train = 10000               # training pictures (at most 60000)
    epochs = 5
    learning_rate = 0.001
    """),
    md(r"""
    **Expected time: about 40 seconds** with the settings above.
    """),
    code("""
    if dataset == "mnist":
        try_train, try_val, classes = P.get_mnist(n_train=n_train, n_val=2000, batch_size=64)
    else:
        try_train, try_val, classes = P.get_fashion_mnist(n_train=n_train, n_val=2000, batch_size=64)

    torch.manual_seed(0)
    cnn_try = build_cnn()
    torch.manual_seed(0)
    fft_try = build_fft_cnn(start)

    history_cnn_try = P.train(cnn_try, try_train, try_val, epochs=epochs, lr=learning_rate, live=False, name="CNN")
    history_fft_try = P.train(fft_try, try_train, try_val, epochs=epochs, lr=learning_rate, live=False,
                              name="FFT-CNN, start " + start)
    P.plot_history({"CNN": history_cnn_try, "FFT-CNN, start " + start: history_fft_try})
    show_fft_filters(fft_try[0], "your FFT-CNN, first layer after training")
    """),
    md(r"""
    <details><summary><b>What we saw</b> (open after you have tried)</summary>

    Best validation accuracy, one run each with seed 0 (so differences below about 0.5 points on MNIST, and
    below a few points with very little data, may be chance):

    | setting | CNN | FFT-CNN, start spectrum | FFT-CNN, start 3x3 |
    |---|---|---|---|
    | 1. `n_train = 300` | 72.3% | 72.6% | 67.3% |
    | 2. `n_train = 60000` | 98.5% | 97.9% | 98.6% |
    | 3. `epochs = 1` | 93.5% | 92.5% | 92.8% |
    | 4. `epochs = 15` | 97.5% | 97.1% | 97.8% |
    | 5. `learning_rate = 0.01` | 97.5% | 96.6% | 97.8% |
    | 6. `dataset = "fashion"` | 85.4% | 86.3% | 86.1% |

    1. **Very little data.** All networks are poor with 300 pictures, and the order between them is not the
       one of section 5: with so few pictures one run says little (remember the spread of 1 to 3 points with
       1 000 pictures). No clear loser.
    2. **A lot of data.** Everybody gains about 1.5 points. The random-spectrum start stays 0.6 points behind:
       six times more data did not close the gap.
    3. **One epoch.** The CNN is ahead by a point; the two FFT-CNNs are level.
    4. **Longer training.** Both FFT-CNNs reach 100.0% on the training pictures, the CNN 99.7%; on the
       validation pictures all three stop improving near 97 to 98%. The FFT-CNNs have the weights to learn
       every training picture by heart. The random-spectrum start comes closer to the CNN (0.4 points) but
       needs three times the epochs for it.
    5. **A larger learning rate.** All three cope. Look at the filters of the 3x3 start afterwards: only 68%
       of them is left inside the central 3 x 3 pixels (98% with the small learning rate). Larger steps carry
       the filters away from the small square, and here that did no harm.
    6. **Fashion-MNIST.** Both FFT-CNNs are a little ahead of the CNN (about 1 point, one run). Clothes differ
       in their overall outline more than in fine strokes, and a filter that sees the whole picture may suit
       that. Repeat it with `repeat_runs` and several seeds before you believe it.
    </details>
    """),

    # ------------------------------------------------------------------ 8
    md(r"""
    ## 8 · What to remember

    * **A picture can be described by its waves instead of its pixels.** The FFT computes that description (the
      spectrum), the inverse FFT goes back, and nothing is lost. Slow waves hold the rough shape, fast waves
      the fine detail.
    * **Convolution theorem: sliding a filter = multiplying spectra.** We checked it with Sobel x: the two
      feature maps agree to a few millionths. A CNN whose filters were rewritten as spectra gave the same
      scores as the original.
    * **An FFT layer learns the spectrum of its filters directly.** That makes it a convolution with a filter
      as large as the picture: every output pixel can depend on every input pixel. It costs 840 weights per
      filter instead of 9, about 1.6 times the time of a training step here, and it works for one picture size only.
    * **The result of the comparison.** Started from the spectra of 3 x 3 filters, the FFT-CNN was as good as
      the CNN or slightly better (97.35% against 96.98%, three seeds). Started from random spectra, it was
      0.7 points worse, and more data or longer training narrowed the gap without closing it.
    * **Why.** The extra freedom was hardly used: after training the 3x3 start was still 98% a 3 x 3 filter,
      and the random start was still noise. Looking at neighbouring pixels is the right idea for pictures; the
      convolution has it built in, the FFT layer has to be given it through its starting point.
    * **How we found out.** One thing changed at a time (twins), several seeds before believing a small
      difference, and a look inside the layers instead of only at the accuracy.

    **Where FFT layers are used in practice.** Not as a replacement for 3 x 3 convolutions: those are cheaper
    and, as we saw, no worse. They are used where a filter must see far in one step. Two examples: *Fourier
    neural operators* (2020), networks that learn to simulate physics such as fluid flow and learn their
    weights per wave as we did here; and *fast Fourier convolution* (2020), used in the inpainting model LaMa
    (2021), where filling a large hole in a photo needs information from the whole picture. Both keep ordinary
    local layers next to the FFT ones.

    ### Words from this notebook

    | word | meaning |
    |---|---|
    | complex number | a pair of ordinary numbers (real and imaginary part) that is added and multiplied as one; the FFT returns one per wave |
    | convolution theorem | sliding a filter over a picture gives the same result as multiplying the spectrum of the picture by the spectrum of the filter |
    | fast wave / slow wave | a wave with thin / wide stripes; fast waves hold the fine detail of a picture, slow waves its rough shape |
    | FFT | fast Fourier transform: the quick algorithm that computes the spectrum of a picture |
    | FFT layer | a layer that takes its input apart into waves, multiplies every wave by a learned weight, and puts the result together again |
    | frequency | how many stripes of a wave fit into the picture |
    | inverse FFT | the way back: from the spectrum to the pixels |
    | overfitting | learning the training pictures by heart: training accuracy keeps rising, validation accuracy does not |
    | receptive field | the part of the picture that one output value of a layer can depend on: 3 x 3 pixels for a 3 x 3 convolution, the whole picture for the FFT layer |
    | spectrum | the description of a picture by its waves: strength and shift of every wave |
    | twins | two networks that start from the same weights and differ in one thing only |
    | wrap-around (circular) border | treating the picture as a tile that repeats: right of the last column comes the first column again |
    """),
]
