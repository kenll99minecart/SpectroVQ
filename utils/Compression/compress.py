# This implementation is inspired from
# https://github.com/facebookresearch/encodec
# which is released under MIT License. Hereafter, the original license:
# MIT License

import typing as tp
import gzip
class BitPacker:
    """Simple bit packer to handle ints with a non standard width, e.g. 10 bits.
    Note that for some bandwidth (1.5, 3), the codebook representation
    will not cover an integer number of bytes.

    Args:
        bits (int): number of bits per value that will be pushed.
        fo (IO[bytes]): file-object to push the bytes to.
    """
    def __init__(self, bits: int, fo: tp.IO[bytes], gzip_compression = None):
        self._current_value = 0
        self._current_bits = 0
        self.bits = bits
        self.fo = gzip.GzipFile(fileobj=fo, mode='wb',compresslevel=gzip_compression) if gzip_compression is not None else fo
        self.use_gzip = gzip_compression is not None

    def push(self, value: int):
        """Push a new value to the stream. This will immediately
        write as many uint8 as possible to the underlying file-object."""
        self._current_value += (value << self._current_bits)
        self._current_bits += self.bits
        while self._current_bits >= 8:
            lower_8bits = self._current_value & 0xff
            self._current_bits -= 8
            self._current_value >>= 8
            #print(bytes([lower_8bits]))
            self.fo.write(bytes([lower_8bits]))

    def flush(self):
        """Flushes the remaining partial uint8, call this at the end
        of the stream to encode."""
        if self._current_bits:
            self.fo.write(bytes([self._current_value]))
            self._current_value = 0
            self._current_bits = 0
        self.fo.flush()
        if self.use_gzip:
            self.fo.close()


class BitUnpacker:
    """BitUnpacker does the opposite of `BitPacker`.

    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
        use_gzip (bool): whether the input stream is gzip compressed.
        """
    def __init__(self, bits: int, fo: tp.IO[bytes], gzip_compression = None):
        self.bits = bits
        self.fo = gzip.GzipFile(fileobj=fo, mode='rb',compresslevel=gzip_compression) if gzip_compression is not None else fo
        self._mask = (1 << bits) - 1
        self._current_value = 0
        self._current_bits = 0

    def pull(self) -> tp.Optional[int]:
        """
        Pull a single value from the stream, potentially reading some
        extra bytes from the underlying file-object.
        Returns `None` when reaching the end of the stream.
        """
        while self._current_bits < self.bits:
            buf = self.fo.read(1)
            if not buf:
                return None
            character = buf[0]
            self._current_value += character << self._current_bits
            self._current_bits += 8

        out = self._current_value & self._mask
        self._current_value >>= self.bits
        self._current_bits -= self.bits
        return out
