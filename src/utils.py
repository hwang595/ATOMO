import blosc

def compress(msg, level=0, name='blosclz'):
    """
    Compress a message.
    """
    if name in {'lz4', 'snappy'}:
        raise ValueError('Do not specify lz4 or snappy. I ran into hard to '
                         'debug issues when I did this. blosclz seems to work')
    # we always use default level for now
    code = blosc.compress(msg, cname=name)
    return bytearray(code)

def decompress(code):
    msg = blosc.decompress(code)
    return msg