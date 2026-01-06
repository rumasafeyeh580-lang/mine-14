"""
PLY Post-Processing Utilities
Provides color correction and refinement for generated PLY files.
"""

import io
from typing import Optional
import numpy as np
from PIL import Image

try:
    import plyfile
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False
    
from logger_config import logger


def correct_ply_colors(
    ply_bytes: bytes, 
    reference_image: Image.Image,
    strength: float = 0.7
) -> bytes:
    """
    Correct PLY vertex colors to better match the reference image.
    
    This applies color statistics transfer from the reference image to the PLY model,
    helping to ensure the 3D model's colors more closely match the original input.
    
    Args:
        ply_bytes: Raw PLY file bytes
        reference_image: Original input image to match colors against
        strength: Correction strength (0.0-1.0), where 1.0 is full correction
        
    Returns:
        Corrected PLY file as bytes
        
    Raises:
        RuntimeError: If plyfile library is not available
    """
    if not PLYFILE_AVAILABLE:
        logger.warning("plyfile library not available, skipping color correction")
        return ply_bytes
    
    try:
        # Parse PLY
        ply_data = plyfile.PlyData.read(io.BytesIO(ply_bytes))
        vertex = ply_data['vertex']
        
        # Check if color data exists
        if not all(field in vertex.data.dtype.names for field in ['red', 'green', 'blue']):
            logger.warning("PLY file has no color data, skipping correction")
            return ply_bytes
        
        # Get reference image color statistics
        ref_array = np.array(reference_image.convert('RGB'))
        ref_mean = ref_array.reshape(-1, 3).mean(axis=0)
        ref_std = ref_array.reshape(-1, 3).std(axis=0)
        
        logger.info(f"Reference image colors - Mean: {ref_mean}, Std: {ref_std}")
        
        # Get PLY vertex colors
        ply_colors = np.column_stack([
            vertex['red'],
            vertex['green'],
            vertex['blue']
        ]).astype(np.float32)
        
        # Calculate PLY color statistics
        ply_mean = ply_colors.mean(axis=0)
        ply_std = ply_colors.std(axis=0)
        
        logger.info(f"PLY colors before correction - Mean: {ply_mean}, Std: {ply_std}")
        
        # Apply color statistics transfer
        # Normalize to zero mean, rescale to match reference std, then shift to reference mean
        corrected = (ply_colors - ply_mean) * (ref_std / (ply_std + 1e-6)) + ref_mean
        
        # Blend with original based on strength
        corrected = ply_colors * (1 - strength) + corrected * strength
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        # Update vertex colors
        vertex['red'] = corrected[:, 0]
        vertex['green'] = corrected[:, 1]
        vertex['blue'] = corrected[:, 2]
        
        corrected_mean = corrected.mean(axis=0)
        logger.success(f"PLY colors after correction - Mean: {corrected_mean}")
        
        # Write corrected PLY to bytes
        output = io.BytesIO()
        ply_data.write(output)
        output.seek(0)
        
        result = output.getvalue()
        logger.info(f"Color correction applied (strength={strength})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during PLY color correction: {e}")
        # Return original on error
        return ply_bytes


def enhance_ply_colors(
    ply_bytes: bytes,
    saturation_boost: float = 1.1,
    brightness_adjustment: float = 1.0
) -> bytes:
    """
    Enhance PLY colors with saturation and brightness adjustments.
    
    Args:
        ply_bytes: Raw PLY file bytes
        saturation_boost: Saturation multiplier (1.0 = no change, >1.0 = more saturated)
        brightness_adjustment: Brightness multiplier (1.0 = no change)
        
    Returns:
        Enhanced PLY file as bytes
    """
    if not PLYFILE_AVAILABLE:
        logger.warning("plyfile library not available, skipping enhancement")
        return ply_bytes
    
    try:
        # Parse PLY
        ply_data = plyfile.PlyData.read(io.BytesIO(ply_bytes))
        vertex = ply_data['vertex']
        
        # Check if color data exists
        if not all(field in vertex.data.dtype.names for field in ['red', 'green', 'blue']):
            logger.warning("PLY file has no color data, skipping enhancement")
            return ply_bytes
        
        # Get colors as RGB
        colors = np.column_stack([
            vertex['red'],
            vertex['green'],
            vertex['blue']
        ]).astype(np.float32) / 255.0
        
        # Convert RGB to HSV for saturation adjustment
        from colorsys import rgb_to_hsv, hsv_to_rgb
        
        enhanced_colors = []
        for r, g, b in colors:
            # Convert to HSV
            h, s, v = rgb_to_hsv(r, g, b)
            
            # Adjust saturation and brightness
            s = min(1.0, s * saturation_boost)
            v = min(1.0, v * brightness_adjustment)
            
            # Convert back to RGB
            r_new, g_new, b_new = hsv_to_rgb(h, s, v)
            enhanced_colors.append([r_new, g_new, b_new])
        
        enhanced_colors = np.array(enhanced_colors) * 255
        enhanced_colors = np.clip(enhanced_colors, 0, 255).astype(np.uint8)
        
        # Update vertex colors
        vertex['red'] = enhanced_colors[:, 0]
        vertex['green'] = enhanced_colors[:, 1]
        vertex['blue'] = enhanced_colors[:, 2]
        
        # Write enhanced PLY to bytes
        output = io.BytesIO()
        ply_data.write(output)
        output.seek(0)
        
        logger.info(f"Color enhancement applied (saturation={saturation_boost}, brightness={brightness_adjustment})")
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error during PLY color enhancement: {e}")
        return ply_bytes

