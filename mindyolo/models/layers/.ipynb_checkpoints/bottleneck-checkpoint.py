from mindspore import nn, ops

from .conv import ConvNormAct, DWConvNormAct


class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), g=(1, 1), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out

    
class Residualblock(nn.Cell):
    def __init__(
        self, c1, c2, k=(1, 3), g=(1, 1), act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernels, groups, expand
        super().__init__()
        self.conv1 = ConvNormAct(c1, c2, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c2, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)

    def construct(self, x):
        out = x + self.conv2(self.conv1(x))
        return out


class C3(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                Bottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class C2f(nn.Cell):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, e=0.5, g=1, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        _c = int(c2 * e)  # hidden channels
        self._c = _c
        self.cv1 = ConvNormAct(c1, 2 * _c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(
            (2 + n) * _c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn
        )  # optional act=FReLU(c2)
        self.m = nn.CellList(
            [
                Bottleneck(_c, _c, shortcut, k=(3, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for i in range(len(self.m)):
            m = self.m[i]
            out = m(y[-1])
            y += (out,)

        return self.cv2(ops.concat(y, axis=1))


class DWBottleneck(nn.Cell):
    # depthwise bottleneck used in yolox nano scale
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = DWConvNormAct(c_, c2, k[1], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out


class DWC3(nn.Cell):
    # depthwise DwC3 used in yolox nano scale, similar as C3
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(DWC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                DWBottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5

class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, sync_bn=False):
        super().__init__(c1, c2, n, shortcut, e)
        c_ = int(c2*e)
        self.m = nn.SequentialCell([
            *(Bottleneck(c_, c_, shortcut, k=(k,k), g=(g,g), e=1.0) for _ in range(n))
        ])
        
    def construct(self, x):
        return self.conv3(ops.cat((self.m(self.conv1(x)), self.conv2(x)), 1))
    

class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, sync_bn=False):
        
        super().__init__(c1=c1, c2=c2, n=n, shortcut=shortcut, g=g, e=e, sync_bn=sync_bn)
        self.m = nn.CellList([
            C3k(self._c, self._c, 2, shortcut, g) if c3k else Bottleneck(self._c, self._c, shortcut, g=(g,g)) for _ in range(n)
        ]) 
        
        
        
        
        
        
        
        
#############################################
class Attention(nn.Cell):
    
    def __init__(self, axis, num_heads=8, attn_ratio=0.5, sync_bn=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = axis // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = axis + nh_kd * 2
        #print(f'Attention:\n    axis={axis}, h={h}')
        self.qkv = ConvNormAct(axis, h, 1, act=False, sync_bn=sync_bn)
        self.proj = ConvNormAct(axis, axis, 1, act=False, sync_bn=sync_bn)
        self.pe = ConvNormAct(axis, axis, 3, 1, g=axis, act=False, sync_bn=sync_bn)

        
        self.softmax = nn.Softmax(axis=-1)
    
    def construct(self, x):
        B, C, H, W = x.shape
        N = H * W
        #print(f'BCHW(x.shape) = {x.shape}, N = {N}')
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], axis=2
        )#############split
        
        
        # q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        #     [self.key_dim, self.key_dim, self.head_dim], axis=2
        # )
        #print(q,k,v)
        attn = (q.swapaxes(-2, -1) @ k) * self.scale
        attn = self.softmax(attn)
        x = (v @ attn.swapaxes(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
    
    
class PSABlock(nn.Cell):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, sync_bn=False) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads, sync_bn=sync_bn)
        self.ffn = nn.SequentialCell(ConvNormAct(c, c*2, 1, sync_bn=sync_bn), ConvNormAct(c*2, c, 1, act=False, sync_bn=sync_bn))
        self.add = shortcut
        
    def construct(self, x):
        x = x+self.attn(x) if self.add else self.attn(x)
        x = x+self.ffn(x) if self.add else self.ffn(x)
        return x
  
    
class C2PSA(nn.Cell):
    def __init__(self, c1, c2, n=1, e=0.5, sync_bn=False):
        super().__init__()
        assert c1==c2
        self.c = int(c1*e)
        self.cv1 = ConvNormAct(c1, 2*self.c, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(2*self.c, c1, 1, sync_bn=sync_bn)
        
        self.m = nn.SequentialCell(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c//64) for _ in range(n)))
        #print(f'for C2PSA: m={self.m}')
        
    def construct(self, x):
        a, b = self.cv1(x).split((self.c, self.c), axis=1)##############split
        #print(a.shape,b.shape)
        b = self.m(b)
        return self.cv2(ops.cat((a,b), 1))
