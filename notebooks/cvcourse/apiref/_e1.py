"""Reference entries for functions first used in E1 (bonus experiments: flattening branches, hand-made filters).

Same format as `_day0.py`. Entries whose key already exists in an earlier file are ignored (the earliest
wording wins), so only functions no earlier notebook explains are here.
"""
from . import E

ENTRIES = [
    E("nn.AvgPool2d", "nn.AvgPool2d(kernel_size, stride=None)",
      """Average pooling: replaces every kernel_size x kernel_size window by the average of its values,
      separately for each channel; it has no parameters. The stride defaults to kernel_size, so
      nn.AvgPool2d(2) halves the height and width: (B, C, 28, 28) -> (B, C, 14, 14). It is the same as a
      convolution whose kernel is filled with 1 / (kernel_size x kernel_size).""",
      """
      >>> x = torch.arange(16.).reshape(1, 1, 4, 4)
      >>> x
      tensor([[[[ 0.,  1.,  2.,  3.],
                [ 4.,  5.,  6.,  7.],
                [ 8.,  9., 10., 11.],
                [12., 13., 14., 15.]]]])
      >>> nn.AvgPool2d(2)(x)                                    # (0 + 1 + 4 + 5) / 4 = 2.5
      tensor([[[[ 2.5000,  4.5000],
                [10.5000, 12.5000]]]])
      >>> nn.AvgPool2d(7)(torch.zeros(1, 1, 28, 28)).shape      # 28 / 7 = 4
      torch.Size([1, 1, 4, 4])
      """,
      note="""nn.MaxPool2d keeps the largest value of each window instead of the average: after max pooling you
      know that a stroke was somewhere in the window, after average pooling how much ink the window holds.""",
      see=["nn.MaxPool2d", "nn.Conv2d"]),
]
