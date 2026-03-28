import zstandard as zstd
import os
import random
import typing as tp
from io import BytesIO


class ZstdAdvancedCompressor:
    """Advanced integer compressor using zstd with delta and zigzag encoding.
    
    This compressor provides high compression ratios for integer sequences by:
    1. Applying delta encoding to reduce entropy
    2. Using zigzag encoding for efficient signed integer representation
    3. Packing integers into 10-bit format for optimal space usage
    4. Using zstd compression without dictionaries for simplicity
    
    Args:
        compression_level (int): Zstd compression level (1-22, default 15).
    """
    
    def __init__(self, compression_level: int = 15):
        self.compression_level = compression_level

    def _zigzag_encode(self, n: int) -> int:
        """Encode integer using zigzag encoding for efficient signed representation."""
        n = int(n)  # Ensure it's a Python int
        return (n << 1) ^ (n >> 31)

    def _zigzag_decode(self, n: int) -> int:
        """Decode zigzag encoded integer."""
        n = int(n)  # Ensure it's a Python int
        return (n >> 1) ^ -(n & 1)

    def _pack_10bit(self, int_list: tp.List[int]) -> bytes:
        """Pack integers into 10-bit format for efficient storage."""
        buffer, bits_in_buffer = 0, 0
        byte_array = bytearray()
        for num in int_list:
            num = int(num)  # Ensure it's a Python int
            # Ensure values fit in 10-bit range (0-1023)
            if num < 0:
                num = 0
            elif num > 1023:
                num = 1023
            buffer = (buffer << 10) | num
            bits_in_buffer += 10
            while bits_in_buffer >= 8:
                bits_in_buffer -= 8
                byte_array.append((buffer >> bits_in_buffer) & 0xFF)
                buffer &= (1 << bits_in_buffer) - 1
        if bits_in_buffer > 0:
            byte_array.append((buffer << (8 - bits_in_buffer)) & 0xFF)
        return bytes(byte_array)

    def _unpack_10bit(self, byte_data: bytes, count: int) -> tp.List[int]:
        """Unpack 10-bit packed integers back to list."""
        result, buffer, bits_in_buffer, ints_read = [], 0, 0, 0
        for byte in byte_data:
            buffer = (buffer << 8) | byte
            bits_in_buffer += 8
            while bits_in_buffer >= 10 and ints_read < count:
                bits_in_buffer -= 10
                result.append((buffer >> bits_in_buffer) & 0x3FF)
                buffer &= (1 << bits_in_buffer) - 1
                ints_read += 1
        return result

    def compress(self, int_list: tp.List[int]) -> bytes:
        """Compress an integer list using delta encoding, zigzag encoding, and zstd.
        
        Args:
            int_list: List of integers to compress.
            
        Returns:
            Compressed bytes data.
        """
        if not int_list:
            return b""
        
        # Convert numpy integers to Python integers if needed
        int_list = [int(x) for x in int_list]
        
        # 1. Delta + ZigZag encoding
        deltas = [int_list[0]]
        for i in range(1, len(int_list)):
            deltas.append(self._zigzag_encode(int_list[i] - int_list[i-1]))
        
        # 2. Find the maximum value to determine required bit width
        max_val = max(abs(val) for val in deltas)
        if max_val <= 1023:
            # Use 10-bit packing for small values
            packed = self._pack_10bit(deltas)
            packing_method = b'\x01'  # 1 = 10-bit packing
        else:
            # For larger values, use 4-byte packing
            packed = b"".join(int(val).to_bytes(4, byteorder='little', signed=True) for val in deltas)
            packing_method = b'\x02'  # 2 = 4-byte packing
        
        # 3. Compress with zstd (no dictionary)
        data_to_compress = packing_method + packed
        cctx = zstd.ZstdCompressor(level=self.compression_level)
        
        return cctx.compress(data_to_compress)

    def decompress(self, compressed_data: bytes, count: int) -> tp.List[int]:
        """Decompress data back to original integer list.
        
        Args:
            compressed_data: Compressed bytes data.
            count: Number of integers in the original list.
            
        Returns:
            Decompressed integer list.
        """
        if not compressed_data or count == 0:
            return []
        
        # 1. Decompress with zstd (no dictionary)
        dctx = zstd.ZstdDecompressor()
        packed_with_method = dctx.decompress(compressed_data)
        
        # 2. Extract packing method from first byte
        if len(packed_with_method) < 1:
            raise ValueError("Invalid compressed data")
        
        packing_method = packed_with_method[0]
        packed = packed_with_method[1:]
        
        # 3. Unpack based on the stored method
        if packing_method == 1:  # 10-bit packing
            zigzag_deltas = self._unpack_10bit(packed, count)
        elif packing_method == 2:  # 4-byte packing
            zigzag_deltas = []
            for i in range(count):
                start = i * 4
                end = start + 4
                if end > len(packed):
                    raise ValueError("Insufficient data for 4-byte unpacking")
                val = int.from_bytes(packed[start:end], byteorder='little', signed=True)
                zigzag_deltas.append(val)
        else:
            raise ValueError(f"Unknown packing method: {packing_method}")
        
        # 4. Inverse Delta + ZigZag
        results = [zigzag_deltas[0]]
        for i in range(1, count):
            results.append(results[i-1] + self._zigzag_decode(zigzag_deltas[i]))
        
        return results


class ZstdBitPacker:
    """BitPacker implementation using zstd compression with advanced encoding.
    
    This class provides the same interface as the original BitPacker but uses
    zstd compression with delta and zigzag encoding for better compression ratios.
    
    Args:
        bits (int): Number of bits per value.
        fo (IO[bytes]): File object to write compressed data to.
        compression_level (int): Zstd compression level (1-22, default 15).
    """
    
    def __init__(self, bits: int, fo: tp.IO[bytes], compression_level: int = 15):
        self.bits = bits
        self.fo = fo
        self.compression_level = compression_level
        self._values = []
        self._advanced_compressor = ZstdAdvancedCompressor(compression_level=compression_level)
        
    def push(self, value: int) -> None:
        """Push a value to be compressed later.
        
        Values are collected and compressed in one batch for better compression.
        """
        self._values.append(value)
        
    def flush(self) -> None:
        """Compress and write all collected values."""
        if not self._values:
            return
            
        # Compress all values at once for better compression
        compressed_data = self._advanced_compressor.compress(self._values)
        
        # Write the count of values followed by compressed data
        count_bytes = len(self._values).to_bytes(4, byteorder='little')
        self.fo.write(count_bytes)
        self.fo.write(compressed_data)
        
        self.fo.flush()
        self._values = []


class ZstdBitUnpacker:
    """BitUnpacker implementation using zstd compression with advanced encoding.
    
    This class provides the same interface as the original BitUnpacker but uses
    zstd compression with delta and zigzag encoding.
    
    Args:
        bits (int): Number of bits per value.
        fo (IO[bytes]): File object to read compressed data from.
        compression_level (int): Zstd compression level (1-22, default 15).
    """
    
    def __init__(self, bits: int, fo: tp.IO[bytes], compression_level: int = 15):
        self.bits = bits
        self.fo = fo
        self._advanced_compressor = ZstdAdvancedCompressor(compression_level=compression_level)
        self._decompressed_values = []
        self._current_index = 0
        self._initialized = False
        
    def _initialize(self) -> None:
        """Read and decompress data from file."""
        if self._initialized:
            return
            
        # Read the count of values
        count_bytes = self.fo.read(4)
        if not count_bytes or len(count_bytes) < 4:
            return
            
        count = int.from_bytes(count_bytes, byteorder='little')
        
        # Read compressed data
        compressed_data = self.fo.read()
        if not compressed_data:
            return
            
        # Decompress
        self._decompressed_values = self._advanced_compressor.decompress(compressed_data, count)
        self._initialized = True
        
    def pull(self) -> tp.Optional[int]:
        """Pull a single value from the decompressed data.
        
        Returns:
            The next integer value, or None if no more values available.
        """
        self._initialize()
        
        if self._current_index >= len(self._decompressed_values):
            return None
            
        value = self._decompressed_values[self._current_index]
        self._current_index += 1
        return value


# Utility functions for integration with existing compression system
def create_zstd_bit_packer(bits: int, fo: tp.IO[bytes], 
                          compression_level: int = 15) -> ZstdBitPacker:
    """Create a ZstdBitPacker instance.
    
    Args:
        bits (int): Number of bits per value.
        fo (IO[bytes]): File object to write to.
        compression_level (int): Zstd compression level (1-22).
        
    Returns:
        ZstdBitPacker: Configured ZstdBitPacker instance.
    """
    return ZstdBitPacker(bits, fo, compression_level)


def create_zstd_bit_unpacker(bits: int, fo: tp.IO[bytes], 
                           compression_level: int = 15) -> ZstdBitUnpacker:
    """Create a ZstdBitUnpacker instance.
    
    Args:
        bits (int): Number of bits per value.
        fo (IO[bytes]): File object to read from.
        compression_level (int): Zstd compression level (1-22).
        
    Returns:
        ZstdBitUnpacker: Configured ZstdBitUnpacker instance.
    """
    return ZstdBitUnpacker(bits, fo, compression_level)


def compress_int_list_zstd(int_list: tp.List[int], 
                          compression_level: int = 15) -> bytes:
    """Compress an integer list using zstd with advanced encoding.
    
    Args:
        int_list: List of integers to compress.
        compression_level: Zstd compression level (1-22).
        
    Returns:
        Compressed bytes data.
    """
    compressor = ZstdAdvancedCompressor(compression_level=compression_level)
    return compressor.compress(int_list)


def decompress_int_list_zstd(compressed_data: bytes, 
                           count: int, 
                           compression_level: int = 15) -> tp.List[int]:
    """Decompress data back to integer list.
    
    Args:
        compressed_data: Compressed bytes data.
        count: Number of integers in original list.
        compression_level: Zstd compression level (1-22).
        
    Returns:
        Decompressed integer list.
    """
    compressor = ZstdAdvancedCompressor(compression_level=compression_level)
    return compressor.decompress(compressed_data, count)
