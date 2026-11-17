"""Checks for P8 (vision transformers). In the notebook: ``from cvcourse import ptests_p8 as T``.

Every check prints one line per property, ``[PASS]`` or ``[FAIL]`` with a plain-language hint, then one summary
line, and never stops the notebook with a traceback: an unfinished step (the ``raise NotImplementedError`` line
still there, or a ``____`` blank left) is reported as "not written yet". The reference versions, written exactly
like the solution cells, live in ``cvcourse.pmodels_p8``.

    T.check_patch_embedding(PatchEmbedding)     step (a)
    T.check_tiny_vit(TinyViT)                   step (b)
    thing = T.yours_or_reference(thing)         yours if it passes its checks, else the reference (+ a note)

A check returns a ``Result`` (true when everything passed; Jupyter shows nothing for it). Checks do not change
PyTorch's random numbers, so the rest of the notebook gives the same results whether a check passed or not.
"""
from __future__ import annotations

import traceback

import torch
import torch.nn as nn

from . import pmodels_p8 as R

__all__ = ["check_patch_embedding", "check_tiny_vit", "yours_or_reference", "explain", "Result"]

NOT_WRITTEN = "not written yet: fill in the ____ blanks and delete the raise lines"


# ============================================================================================== reporting
class Result:
    """What a check returns: true when every line passed. Prints nothing itself in Jupyter."""

    def __init__(self, passed: bool, n_pass: int, n_fail: int):
        self.passed, self.n_pass, self.n_fail = passed, n_pass, n_fail

    def __bool__(self):
        return self.passed

    def __repr__(self):
        return f"Result(passed={self.passed}, {self.n_pass} passed, {self.n_fail} failed)"

    def _ipython_display_(self):
        pass                                 # the lines are already printed


class _Lines:
    """Collects [PASS] / [FAIL] lines; prints them only when verbose."""

    def __init__(self, title: str, verbose: bool):
        self.verbose = verbose
        self.n_pass = 0
        self.n_fail = 0
        if verbose:
            print(title)

    def ok(self, name: str):
        self.n_pass += 1
        if self.verbose:
            print(f"[PASS]  {name}")

    def fail(self, name: str, hint: str):
        self.n_fail += 1
        if self.verbose:
            print(f"[FAIL]  {name}")
            if hint:
                print(f"        hint: {hint}")

    def check(self, condition: bool, name: str, hint: str = "") -> bool:
        if condition:
            self.ok(name)
        else:
            self.fail(name, hint)
        return bool(condition)

    def done(self, step: str, what: str) -> Result:
        passed = self.n_fail == 0 and self.n_pass > 0
        if self.verbose:
            if passed:
                print(f"=> all {self.n_pass} checks passed: step {step} is done. {what}")
            else:
                total = self.n_pass + self.n_fail
                print(f"=> {self.n_fail} of {total} checks failed: fix the [FAIL] lines in your step {step} cell, "
                      f"run it, then run this check again.")
        return Result(passed, self.n_pass, self.n_fail)


def _first_line(e: BaseException) -> str:
    text = str(e).strip()
    if not text:
        return type(e).__name__
    return text.splitlines()[0]


def _where(e: BaseException) -> str:
    """' (in forward, line 7 of its cell)': the last place in the student's own code."""
    mine = None
    for frame in traceback.extract_tb(e.__traceback__):
        f = frame.filename.replace("\\", "/")
        if "/torch/" in f or "/cvcourse/" in f or "site-packages" in f or "/lib/python" in f.lower():
            continue
        mine = frame
    if mine is None:
        return ""
    if mine.name == "<module>":
        return f" (line {mine.lineno} of the cell)"
    return f" (in {mine.name}, line {mine.lineno} of its cell)"


def explain(e: BaseException) -> str:
    """A plain-language reason for an error raised by the student's code."""
    msg = _first_line(e)
    where = _where(e)
    if isinstance(e, NotImplementedError):
        return NOT_WRITTEN
    if isinstance(e, NameError):
        if "'____'" in msg:
            return ("a ____ blank is still empty" + where + ": replace every ____ with what its # YOUR CODE comment "
                    "asks for")
        return (f"Python does not know a name ({msg}){where}: check the spelling, and run all the cells above this "
                f"one first")
    if isinstance(e, AttributeError):
        return (f"{msg}{where}: keep the names of the template (self.to_tokens, self.class_token, self.position, "
                f"self.embedding, self.blocks, self.norm, self.head) and run the cells above first")
    if isinstance(e, TypeError):
        if "cat()" in msg:
            return "torch.cat takes ONE list of tensors and the axis: torch.cat([class_tokens, x], dim=1)" + where
        if "Parameter" in msg or "parameter" in msg:
            return (f"{msg}{where}: nn.Parameter wraps one tensor, e.g. nn.Parameter(torch.zeros(1, 1, dim))")
        return f"Python stopped with a TypeError{where}: {msg}. Compare the brackets and arguments with the table"
    if isinstance(e, (RuntimeError, ValueError, AssertionError)):
        if "expanded size" in msg or "expand" in msg:
            return ("the class token has the wrong shape" + where + ": it must be 1 x 1 x dim, "
                    "nn.Parameter(torch.zeros(1, 1, dim))")
        if "Sizes of tensors must match except in dimension 1" in msg:
            return ("the class token and the patch tokens do not have the same number of numbers per token" + where
                    + ": class_tokens is batch x 1 x dim and x is batch x patches x dim; keep the flatten(2) and "
                    "transpose(1, 2) lines and use dim maps in the convolution")
        if "Sizes of tensors must match except in dimension" in msg:
            return ("torch.cat joins along the wrong axis" + where + ": use dim=1, the axis of the tokens "
                    "(axis 0 counts the pictures of the batch)")
        if "must match the size of tensor" in msg:
            return (f"two tensors of different sizes are added{where} ({msg}). self.position needs one row per "
                    f"token: number_of_patches + 1 rows (the + 1 is the class token), and the class token must be "
                    f"put in front before the position is added")
        if "divisible by num_heads" in msg or "embed_dim must be divisible" in msg:
            return (f"{msg}{where}: dim must be a multiple of heads (dim 32 with 4 heads gives each head 8 numbers)")
        if "mat1 and mat2" in msg:
            return (f"a linear layer receives a different number of numbers than it was built for{where}: {msg}. "
                    f"The head reads the class token, dim numbers: nn.Linear(dim, num_classes)")
        if "channels, but got" in msg or ("expected input" in msg and "channels" in msg):
            return (f"the convolution receives a different number of channels than it was built for{where}: {msg}. "
                    f"Use in_channels as its first number")
        if "too small" in msg or "Kernel size can't be greater" in msg:
            return f"the picture is smaller than the filter{where}: {msg}"
    return f"Python stopped with {type(e).__name__}{where}: {msg}"


def _try(fn, *args, **kwargs):
    """(value, None) or (None, plain-language reason)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:  # noqa: BLE001  (anything the student's code raises becomes a hint)
        return None, explain(e)


def _shape(t) -> str:
    return " x ".join(str(int(v)) for v in t)


def _params(m: nn.Module) -> int:
    total = 0
    for p in m.parameters():
        total += p.numel()
    return total


def _run(model: nn.Module, x: torch.Tensor):
    model.eval()
    with torch.no_grad():
        return model(x)


def _with_rng(fn):
    """Run fn() without changing PyTorch's random numbers for the rest of the notebook; never raise."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  (a check must never end in a traceback)
            print(f"[FAIL]  the check could not finish: {explain(e)}")
            return Result(False, 0, 1)


def _is_registered(module: nn.Module, name: str) -> bool:
    for key, _ in module.named_parameters(recurse=False):
        if key == name:
            return True
    return False


def _pair(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)


# ============================================================================================== step (a)
def _expected_tokens(emb: nn.Module, x: torch.Tensor, position: bool = True, class_first: bool = True):
    """What the embedding should give, computed from the student's own layer and parameters."""
    t = emb.to_tokens(x)
    t = t.flatten(2)
    t = t.transpose(1, 2)
    cls = emb.class_token.expand(t.shape[0], -1, -1)
    if class_first:
        t = torch.cat([cls, t], dim=1)
    else:
        t = torch.cat([t, cls], dim=1)
    if position:
        t = t + emb.position
    return t


def _conv_problems(conv, cin: int, dim: int, p: int) -> list:
    out = []
    if conv.in_channels != cin:
        out.append(f"in channels is {conv.in_channels}, expected {cin} (in_channels: 1 for a grey picture)")
    if conv.out_channels != dim:
        out.append(f"out channels is {conv.out_channels}, expected {dim} (dim: one map per number of a token)")
    if _pair(conv.kernel_size) != (p, p):
        out.append(f"kernel_size is {_pair(conv.kernel_size)[0]}, expected {p} (patch_size: one filter covers one "
                   f"patch)")
    if _pair(conv.stride) != (p, p):
        out.append(f"stride is {_pair(conv.stride)[0]}, expected {p}: with stride=patch_size the filter jumps from "
                   f"patch to patch; without it, it visits every pixel position and the patches overlap")
    if isinstance(conv.padding, str) or _pair(conv.padding) != (0, 0):
        out.append(f"padding is {conv.padding}, expected 0 (leave it out)")
    return out


def _patch_embedding(cls, verbose: bool) -> Result:
    lines = _Lines("Checking step (a): PatchEmbedding(image_size=28, patch_size=7, in_channels=1, dim=32)", verbose)
    emb, err = _try(cls, image_size=28, patch_size=7, in_channels=1, dim=32)
    if not lines.check(err is None, "PatchEmbedding(image_size=28, patch_size=7, in_channels=1, dim=32) can be built",
                       err):
        return lines.done("(a)", "")
    conv = getattr(emb, "to_tokens", None)
    conv_ok = False
    if isinstance(conv, nn.Conv2d):
        problems = _conv_problems(conv, 1, 32, 7)
        conv_ok = lines.check(not problems, "self.to_tokens: nn.Conv2d(1, 32, kernel_size=7, stride=7), 1,600 "
                              "parameters", "; ".join(problems))
    else:
        found = type(conv).__name__ if conv is not None else "nothing"
        lines.fail("self.to_tokens: nn.Conv2d(1, 32, kernel_size=7, stride=7), 1,600 parameters",
                   f"self.to_tokens is {found}: the table asks for nn.Conv2d(in_channels, dim, "
                   f"kernel_size=patch_size, stride=patch_size)")
    shapes_ok = True
    for name, want, words in (("class_token", (1, 1, 32), "1 x 1 x 32: one row of dim numbers"),
                              ("position", (1, 17, 32), "1 x 17 x 32: one row per token, 16 patches + 1 class token")):
        value = getattr(emb, name, None)
        title = f"self.{name}: a learned nn.Parameter of {words}"
        if not isinstance(value, torch.Tensor):
            lines.fail(title, f"self.{name} is {type(value).__name__ if value is not None else 'missing'}")
            shapes_ok = False
            continue
        if not _is_registered(emb, name):
            lines.fail(title, f"self.{name} is a plain tensor, so the network would never learn it: wrap it, "
                              f"nn.Parameter(torch....)")
            shapes_ok = False
            continue
        if tuple(value.shape) != want:
            hint = f"its shape is {_shape(value.shape)}, expected {_shape(want)}"
            if name == "position" and tuple(value.shape) == (1, 16, 32):
                hint += ": one row too few. The class token needs a position row too: number_of_patches + 1"
            if name == "class_token" and value.dim() != 3:
                hint += ": write all three numbers, torch.zeros(1, 1, dim)"
            lines.fail(title, hint)
            shapes_ok = False
            continue
        lines.ok(title)
    x = torch.randn(2, 1, 28, 28)
    out, err = _try(_run, emb, x)
    if err and not (conv_ok and shapes_ok):
        err = "fix the [FAIL] lines above first; then: " + err
    if not lines.check(err is None and isinstance(out, torch.Tensor), "forward runs on a batch of 2 pictures, "
                       "2 x 1 x 28 x 28", err or "forward must end with: return x"):
        return lines.done("(a)", "")
    hint = f"the output is {_shape(out.shape)}, expected 2 x 17 x 32"
    if tuple(out.shape) == (2, 16, 32):
        hint += ": the class token is missing; put it in front with torch.cat([class_tokens, x], dim=1)"
    elif out.dim() == 4:
        hint += ": forward stops after the convolution; fill in the lines after it"
    shape_ok = lines.check(tuple(out.shape) == (2, 17, 32), "the output is 2 x 17 x 32: per picture 17 tokens "
                           "(1 class token + 16 patches) of 32 numbers", hint)
    if shape_ok and shapes_ok and isinstance(conv, nn.Conv2d):
        with torch.no_grad():
            full = _expected_tokens(emb, x)
            no_pos = _expected_tokens(emb, x, position=False)
            last = _expected_tokens(emb, x, class_first=False)
            last_no_pos = _expected_tokens(emb, x, position=False, class_first=False)
        # the place of the class token: row 0, with or without its position row (that is the next line's job)
        first_ok = torch.allclose(out[:, 0], full[:, 0], atol=1e-5) or torch.allclose(out[:, 0], no_pos[:, 0],
                                                                                        atol=1e-5)
        hint = "row 0 must be the class token (plus its position row)"
        if torch.allclose(out[:, -1], last[:, -1], atol=1e-5) or torch.allclose(out[:, -1], last_no_pos[:, -1],
                                                                                 atol=1e-5):
            hint = "the class token is at the end: put it first, torch.cat([class_tokens, x], dim=1)"
        lines.check(first_ok, "row 0 of every picture is the class token: the class token comes first", hint)
        all_ok = torch.allclose(out, full, atol=1e-5)
        hint = "the numbers differ from what the table describes: compare forward with the table, line by line"
        if not first_ok:
            hint = "fix the place of the class token first (the line above)"
        elif torch.allclose(out, no_pos, atol=1e-5):
            hint = "the position rows are not added: the last line is x = x + self.position"
        elif torch.allclose(out, no_pos + 2 * emb.position, atol=1e-5):
            hint = "the position rows are added twice"
        lines.check(all_ok, "rows 1 to 16 are the patches in order (left to right, top to bottom), and every row "
                    "has its position row added", hint)
    want = 1600 + 32 + 17 * 32
    got = _params(emb)
    lines.check(got == want, f"{want:,} parameters: 1,600 (convolution) + 32 (class token) + 544 (position, 17 x 32)",
                f"yours has {got:,}")
    other, err = _try(cls, image_size=32, patch_size=4, in_channels=3, dim=16)
    title = ("also right with other numbers: PatchEmbedding(image_size=32, patch_size=4, in_channels=3, dim=16) "
             "turns a 3 x 32 x 32 colour picture into 65 tokens of 16")
    names = ("Use the names image_size, patch_size, in_channels, dim and number_of_patches, not the numbers 28, 7, "
             "1, 32, 16 or 17")
    if err:
        lines.fail(title, err + ". " + names)
    else:
        y, err = _try(_run, other, torch.randn(2, 3, 32, 32))
        ok = err is None and isinstance(y, torch.Tensor) and tuple(y.shape) == (2, 65, 16)
        if err:
            hint = err + ". " + names
        else:
            hint = (f"the output is {_shape(y.shape) if isinstance(y, torch.Tensor) else type(y).__name__}, expected "
                    f"2 x 65 x 16. " + names)
        lines.check(ok, title, hint)
    return lines.done("(a)", "Your PatchEmbedding turns a picture into tokens.")


def check_patch_embedding(PatchEmbedding) -> Result:
    """Step (a): the convolution, the class token and the position rows; 17 tokens of 32 numbers per digit."""
    return _with_rng(lambda: _patch_embedding(PatchEmbedding, True))


# ============================================================================================== step (b)
def _recorder(calls: list, name: str):
    def hook(module, inputs, output):
        calls.append(name)
    return hook


def _record_calls(net: nn.Module, x: torch.Tensor) -> list:
    calls = []
    handles = []
    for name in ("embedding", "blocks", "norm", "head"):
        part = getattr(net, name, None)
        if isinstance(part, nn.Module):
            handles.append(part.register_forward_hook(_recorder(calls, name)))
    try:
        _run(net, x)
    finally:
        for h in handles:
            h.remove()
    return calls


def _blocks_of(net: nn.Module) -> list:
    out = []
    blocks = getattr(net, "blocks", None)
    if not isinstance(blocks, nn.Module):
        return out
    for m in blocks.children():
        out.append(m)
    return out


def _variants(net: nn.Module, x: torch.Tensor) -> dict:
    """Scores the student's parts would give for the usual ways of reading the tokens."""
    with torch.no_grad():
        tokens = net.embedding(x)
        tokens = net.blocks(tokens)
        out = {
            "row 0": net.head(net.norm(tokens[:, 0])),
            "last row": net.head(net.norm(tokens[:, -1])),
            "row 1": net.head(net.norm(tokens[:, 1])),
            "average": net.head(net.norm(tokens.mean(dim=1))),
            "no norm": net.head(tokens[:, 0]),
        }
    return out


def _tiny_vit(cls, verbose: bool) -> Result:
    lines = _Lines("Checking step (b): TinyViT() with its defaults: patch_size=7, dim=32, depth=2, heads=4, "
                   "num_classes=10", verbose)
    net, err = _try(cls)
    if not lines.check(err is None, "TinyViT() can be built", err):
        return lines.done("(b)", "")
    emb = getattr(net, "embedding", None)
    emb_ok = isinstance(emb, nn.Module) and isinstance(getattr(emb, "to_tokens", None), nn.Conv2d)
    hint = "self.embedding must be PatchEmbedding(image_size, patch_size, 1, dim)"
    if emb_ok:
        conv = emb.to_tokens
        problems = _conv_problems(conv, 1, 32, 7)
        emb_ok = not problems
        if problems:
            hint = ("self.embedding was built with other numbers: " + "; ".join(problems) +
                    ". Write PatchEmbedding(image_size, patch_size, 1, dim)")
    lines.check(emb_ok, "self.embedding: PatchEmbedding(image_size, patch_size, 1, dim), the step (a) layer", hint)
    blocks = _blocks_of(net)
    good = 0
    for b in blocks:
        attn = getattr(b, "self_attn", None)
        if isinstance(b, nn.TransformerEncoderLayer) and attn is not None and attn.embed_dim == 32 and \
                attn.num_heads == 4:
            good += 1
    hint = f"found {len(blocks)} block(s), {good} of them transformer_block(32, 4)"
    if len(blocks) == 2 and good < 2:
        hint += ": inside the loop write transformer_block(dim, heads), with the names, not numbers"
    lines.check(len(blocks) == 2 and good == 2, "self.blocks: depth = 2 transformer blocks, "
                "transformer_block(dim, heads) with dim 32 and 4 heads", hint)
    head = getattr(net, "head", None)
    if isinstance(head, nn.Linear):
        lines.check(head.in_features == 32 and head.out_features == 10, "self.head: nn.Linear(32, 10), from the "
                    "class token's 32 numbers to 10 scores", f"found nn.Linear({head.in_features}, "
                    f"{head.out_features}); the table asks for nn.Linear(dim, num_classes)")
    else:
        lines.fail("self.head: nn.Linear(32, 10), from the class token's 32 numbers to 10 scores",
                   f"self.head is {type(head).__name__ if head is not None else 'missing'}")
    norm = getattr(net, "norm", None)
    lines.check(isinstance(norm, nn.LayerNorm), "self.norm: nn.LayerNorm(dim) (given)",
                "keep the given line self.norm = nn.LayerNorm(dim)")
    x = torch.randn(2, 1, 28, 28)
    out, err = _try(_run, net, x)
    if not lines.check(err is None and isinstance(out, torch.Tensor), "forward runs on a batch of 2 pictures, "
                       "2 x 1 x 28 x 28", err or "forward must end with: return scores"):
        return lines.done("(b)", "")
    hint = f"the output is {_shape(out.shape)}, expected 2 x 10"
    if out.dim() == 3:
        hint += ": the head gets all 17 tokens of each picture; give it only the class token, tokens[:, 0]"
    elif out.dim() == 2 and out.shape[0] == 17:
        hint += (": tokens[0] is the first PICTURE (all its 17 tokens); row 0 of every picture is tokens[:, 0] "
                 "(':' = every picture, 0 = row 0)")
    lines.check(tuple(out.shape) == (2, 10), "the output is 2 x 10: 10 scores for each of the 2 pictures", hint)
    calls, err = _try(_record_calls, net, x)
    lines.check(calls == ["embedding", "blocks", "norm", "head"], "forward uses the parts in order, each once: "
                "embedding, blocks, norm, head", f"your forward uses: {', '.join(calls or ['nothing'])}")
    parts_ok = emb_ok and len(blocks) == 2 and isinstance(head, nn.Linear) and isinstance(norm, nn.LayerNorm)
    if parts_ok and tuple(out.shape) == (2, 10):
        variants, err = _try(_variants, net, x)
        if err:
            lines.fail("the scores are read from the class token, row 0", err)
        else:
            hint = "the scores differ from head(norm(tokens[:, 0]))"
            if torch.allclose(out, variants["last row"], atol=1e-5):
                hint = "you read the last row (the last patch); the class token is row 0: tokens[:, 0]"
            elif torch.allclose(out, variants["row 1"], atol=1e-5):
                hint = "you read row 1, which is patch 0; the class token is row 0: tokens[:, 0]"
            elif torch.allclose(out, variants["average"], atol=1e-5):
                hint = ("you average all tokens (some ViTs do that); ours reads the class token, row 0: "
                        "tokens[:, 0]")
            elif torch.allclose(out, variants["no norm"], atol=1e-5):
                hint = "the given line class_token = self.norm(class_token) is missing"
            lines.check(torch.allclose(out, variants["row 0"], atol=1e-5), "the scores are read from the class "
                        "token: head(norm(tokens[:, 0]))", hint)
    want = _params(R.TinyViT())
    got = _params(net)
    lines.check(got == want, f"{want:,} parameters: embedding 2,176 + 2 blocks of 8,544 + norm 64 + head 330",
                f"yours has {got:,}")
    earlier_ok = lines.n_fail == 0
    other, err = _try(cls, patch_size=4, dim=16, depth=3, heads=2, num_classes=5)
    title = "also right with other numbers: TinyViT(patch_size=4, dim=16, depth=3, heads=2, num_classes=5)"
    if err:
        lines.fail(title, err)
    else:
        y, err = _try(_run, other, torch.randn(2, 1, 28, 28))
        want_other = _params(R.TinyViT(patch_size=4, dim=16, depth=3, heads=2, num_classes=5))
        ok = err is None and isinstance(y, torch.Tensor) and tuple(y.shape) == (2, 5) and _params(other) == want_other
        hint = err
        if not hint:
            hint = (f"output {_shape(y.shape) if isinstance(y, torch.Tensor) else type(y).__name__} and "
                    f"{_params(other):,} parameters, expected 2 x 5 and {want_other:,}. ")
            if earlier_ok:
                hint += "Use the names (image_size, patch_size, dim, depth, heads, num_classes), not numbers"
            else:
                hint += "Fix the [FAIL] lines above first."
        lines.check(ok, title + ": 2 x 5 scores", hint)
    return lines.done("(b)", "You have built a vision transformer.")


def check_tiny_vit(TinyViT) -> Result:
    """Step (b): the whole tiny ViT, part by part, and the class token read at the end."""
    return _with_rng(lambda: _tiny_vit(TinyViT, True))


# ============================================================================================== never stuck
_STEPS = {
    "PatchEmbedding": (_patch_embedding, R.PatchEmbedding, "(a)"),
    "TinyViT": (_tiny_vit, R.TinyViT, "(b)"),
}


def yours_or_reference(yours, name: str | None = None):
    """Your class if it passes all its checks; otherwise the course's reference version, with a note.

    So that the rest of the notebook works while a step is unfinished: use it as
    ``PatchEmbedding = T.yours_or_reference(PatchEmbedding)``. Fix your cell later, run it, its check and this
    line again.
    """
    name = name or getattr(yours, "__name__", "")
    if name not in _STEPS:
        raise ValueError(f"yours_or_reference knows {', '.join(_STEPS)}; got {name!r}")
    check, reference, step = _STEPS[name]
    if yours is reference:
        print(f"note: still using the reference {name}: once your step {step} passes, run your cell, its check and "
              f"this cell again")
        return reference
    result = _with_rng(lambda: check(yours, False))
    if result:
        print(f"using your {name} (it passes all its checks)")
        return yours
    print(f"note: using the reference {name} so you can continue; come back to step {step} later "
          f"(your version does not pass its checks yet)")
    return reference
