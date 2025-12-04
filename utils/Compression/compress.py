# This implementation is inspired from
# https://github.com/facebookresearch/encodec
# which is released under MIT License. Hereafter, the original license:
# MIT License

import typing as tp
import gzip
import zlib
import io

class ZlibCompressedFile:
    """A file-like object that compresses data using zlib before writing to the underlying file
    or decompresses data when reading.
    
    Args:
        fileobj (IO[bytes]): The underlying file object to read from or write to.
        compression_level (int, optional): Compression level (0-9), where 9 is highest compression.
                                           Default is 6 (good balance between speed and compression).
        mode (str, optional): 'r' for read mode, 'w' for write mode. Default is None, which 
                             determines the mode based on whether a compressor or decompressor is created.
    """
    def __init__(self, fileobj: tp.IO[bytes], compression_level: int = 6, mode: str = None):
        self.fileobj = fileobj
        self.mode = mode or 'w'  # Default to write mode if not specified
        self.compressor = None
        self.decompressor = None
        self._buffer = b""
        
        # Initialize appropriate objects based on mode
        if self.mode == 'w':
            self.compressor = zlib.compressobj(compression_level)
        else:  # 'r' mode
            self.decompressor = zlib.decompressobj()
        
    def write(self, data: bytes) -> int:
        if self.mode != 'w':
            raise IOError("File not open for writing")
            
        if self.compressor is None:
            self.compressor = zlib.compressobj()
            
        compressed = self.compressor.compress(data)
        if compressed:
            return self.fileobj.write(compressed)
        return 0
        
    def read(self, size: int = -1) -> bytes:
        """Read and decompress data from the file.
        
        Args:
            size (int): Number of bytes to read after decompression.
                        If -1 (default), read all available data.
        """
        if self.mode != 'r':
            raise IOError("File not open for reading")
            
        if self.decompressor is None:
            self.decompressor = zlib.decompressobj()
            
        if size < 0:
            # Read all available data
            compressed = self.fileobj.read()
            if not compressed:
                return b''
            
            # Decompress all data
            try:
                decompressed = self.decompressor.decompress(compressed)
                decompressed += self.decompressor.flush()
                return decompressed
            except zlib.error:
                # Handle potential zlib errors by returning what we have
                return b''
        
        # If we already have enough data in the buffer, return it
        if len(self._buffer) >= size:
            result = self._buffer[:size]
            self._buffer = self._buffer[size:]
            return result
            
        # Read compressed data in chunks until we have enough decompressed data
        result = self._buffer
        self._buffer = b""
        
        while len(result) < size:
            chunk = self.fileobj.read(4096)  # Read in reasonable chunks
            if not chunk:
                # End of file reached
                if self.decompressor:
                    try:
                        result += self.decompressor.flush()
                    except zlib.error:
                        pass  # Ignore flush errors at EOF
                return result
                
            try:
                decompressed = self.decompressor.decompress(chunk)
                if len(result) + len(decompressed) <= size:
                    # Still need more data
                    result += decompressed
                else:
                    # We have enough data now
                    needed = size - len(result)
                    result += decompressed[:needed]
                    self._buffer = decompressed[needed:]
                    break
            except zlib.error:
                # Handle decompress errors by returning what we have so far
                return result
                
        return result
        
    def flush(self) -> None:
        if self.mode == 'w' and self.compressor:
            try:
                remaining = self.compressor.flush()
                if remaining:
                    self.fileobj.write(remaining)
            except zlib.error:
                # Handle potential zlib errors during flush
                pass
        self.fileobj.flush()
        
    def close(self) -> None:
        try:
            self.flush()
        except:
            pass  # Ignore errors on flush during close
        self.fileobj.close()
class BitPacker:
    """Simple bit packer to handle ints with a non standard width, e.g. 10 bits.
    Note that for some bandwidth (1.5, 3), the codebook representation
    will not cover an integer number of bytes.

    Args:
        bits (int): number of bits per value that will be pushed.
        fo (IO[bytes]): file-object to push the bytes to.
        gzip_compression (int, optional): Level of gzip compression (1-9), None for no compression.
        zlib_compression (int, optional): Level of zlib compression (1-9), None for no compression.
    """
    """Simple bit packer to handle ints with a non standard width, e.g. 10 bits.
    Note that for some bandwidth (1.5, 3), the codebook representation
    will not cover an integer number of bytes.

    Args:
        bits (int): number of bits per value that will be pushed.
        fo (IO[bytes]): file-object to push the bytes to.
        gzip_compression (int, optional): Level of gzip compression (1-9), None for no compression.
        zlib_compression (int, optional): Level of zlib compression (1-9), None for no compression.
    """
    def __init__(self, bits: int, fo: tp.IO[bytes], gzip_compression = None, zlib_compression = None):
        self._current_value = 0
        self._current_bits = 0
        self.bits = bits
        
        if gzip_compression is not None and zlib_compression is not None:
            raise ValueError("Cannot use both gzip and zlib compression simultaneously")
        
        if gzip_compression is not None:
            self.fo = gzip.GzipFile(fileobj=fo, mode='wb', compresslevel=gzip_compression)
            self.compression_type = 'gzip'
        elif zlib_compression is not None:
            self.fo = ZlibCompressedFile(fo, compression_level=zlib_compression, mode='w')
            self.compression_type = 'zlib'
        else:
            self.fo = fo
            self.compression_type = None

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
        
        try:
            self.fo.flush()
            if self.compression_type in ['gzip', 'zlib']:
                self.fo.close()
        except Exception as e:
            # Handle potential flush errors - log them if needed
            import sys
            print(f"Warning: Error during flush: {str(e)}", file=sys.stderr)


class BitUnpacker:
    """BitUnpacker does the opposite of `BitPacker`.

    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
        gzip_compression (int, optional): Level of gzip compression used, None if no gzip compression.
        zlib_compression (int, optional): Level of zlib compression used, None if no zlib compression.
    """
    """BitUnpacker does the opposite of `BitPacker`.

    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
        gzip_compression (int, optional): Level of gzip compression used, None if no gzip compression.
        zlib_compression (int, optional): Level of zlib compression used, None if no zlib compression.
    """
    def __init__(self, bits: int, fo: tp.IO[bytes], gzip_compression = None, zlib_compression = None):
        self.bits = bits
        
        if gzip_compression is not None and zlib_compression is not None:
            raise ValueError("Cannot use both gzip and zlib compression simultaneously")
            
        if gzip_compression is not None:
            self.fo = gzip.GzipFile(fileobj=fo, mode='rb')
        elif zlib_compression is not None:
            self.fo = ZlibCompressedFile(fo, mode='r')
        else:
            self.fo = fo
            
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


def create_compressed_bit_packer(bits: int, fo: tp.IO[bytes], 
                                compression_type: str = 'zlib', 
                                compression_level: int = 6) -> BitPacker:
    """Create a BitPacker with the specified compression.
    
    Args:
        bits (int): Number of bits per value that will be pushed.
        fo (IO[bytes]): File-object to push the bytes to.
        compression_type (str): Type of compression to use ('zlib', 'gzip', or None).
        compression_level (int): Compression level (1-9), where 9 is highest compression.
        
    Returns:
        BitPacker: A configured BitPacker instance with the specified compression.
    """
    if compression_type is None:
        return BitPacker(bits, fo)
    
    if compression_type.lower() == 'zlib':
        return BitPacker(bits, fo, zlib_compression=compression_level)
    
    if compression_type.lower() == 'gzip':
        return BitPacker(bits, fo, gzip_compression=compression_level)
    
    raise ValueError(f"Unsupported compression type: {compression_type}. Use 'zlib', 'gzip', or None.")


def create_compressed_bit_unpacker(bits: int, fo: tp.IO[bytes],
                                  compression_type: str = 'zlib') -> BitUnpacker:
    """Create a BitUnpacker with the specified compression.
    
    Args:
        bits (int): Number of bits of the values to decode.
        fo (IO[bytes]): File-object to read compressed data from.
        compression_type (str): Type of compression used ('zlib', 'gzip', or None).
        
    Returns:
        BitUnpacker: A configured BitUnpacker instance with the specified compression.
    """
    if compression_type is None:
        return BitUnpacker(bits, fo)
    
    if compression_type.lower() == 'zlib':
        return BitUnpacker(bits, fo, zlib_compression=1)
    
    if compression_type.lower() == 'gzip':
        return BitUnpacker(bits, fo, gzip_compression=1)
    
    raise ValueError(f"Unsupported compression type: {compression_type}. Use 'zlib', 'gzip', or None.")


def zlib_compress(data: bytes, level: int = 6) -> bytes:
    """Compress data using zlib.
    
    Args:
        data (bytes): Data to compress.
        level (int): Compression level (0-9), where 9 is highest compression.
    
    Returns:
        bytes: Compressed data.
    """
    return zlib.compress(data, level)


def zlib_decompress(data: bytes) -> bytes:
    """Decompress zlib-compressed data.
    
    Args:
        data (bytes): Compressed data to decompress.
    
    Returns:
        bytes: Decompressed data.
    """
    try:
        return zlib.decompress(data)
    except zlib.error as e:
        raise ValueError(f"Error decompressing data: {str(e)}")
