from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional

from PIL import Image
import numpy as np
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.ply_post_processor import correct_ply_colors
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)


def get_optimal_background_color(image: Image.Image) -> str:
    """
    Select optimal background color for maximum contrast with the object.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Background color description string for the prompt
    """
    try:
        # Convert to numpy array and calculate average brightness
        arr = np.array(image.convert('RGB'))
        brightness = arr.mean()
        
        # Select contrasting background based on object brightness
        if brightness > 128:
            # Dark object -> Light background
            return "solid light gray (#B0B0B0) background"
        else:
            # Light object -> Dark background
            return "solid dark gray (#505050) background"
    except Exception as e:
        logger.warning(f"Could not determine optimal background: {e}, using default")
        return "solid medium gray background"


def create_view_prompt(view_name: str, background_color: str) -> str:
    """
    Create optimized prompt for specific view with color preservation.
    
    Args:
        view_name: The view angle (e.g., "left three-quarters view")
        background_color: Background color description
        
    Returns:
        Complete prompt string
    """
    prompt = (
        f"CRITICAL REQUIREMENTS: Preserve exact original RGB colors and textures precisely. "
        f"Show this object in {view_name} with complete visibility. "
        f"Background: {background_color}. "
        f"Maintain all material properties (metallic sheen, matte finish, glossy surface, transparency). "
        f"Remove all watermarks, text overlays, and background clutter. "
        f"Maximize detail sharpness and clarity. "
        f"Keep original proportions, dimensions, and geometric features accurate. "
        f"Ensure proper lighting - even, neutral, studio-quality illumination."
    )
    return prompt


class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()

        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""

        temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_image_bytes, seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file

        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerateRequest(
            prompt_image=image_base64, prompt_type="image", seed=seed
        )

        # Generate
        response = await self.generate_gs(request)

        # Return binary PLY
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")

        return response.ply_file_base64  # bytes

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # Determine optimal background color for maximum contrast
        background_color = get_optimal_background_color(image)
        logger.info(f"Selected background: {background_color}")

        # 1. Edit the image using Qwen Edit - Left view
        logger.info("Generating left three-quarters view...")
        image_edited = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt=create_view_prompt("left three-quarters view", background_color),
        )

        # 2. Remove background
        image_without_background = self.rmbg.remove_background(image_edited)

        # 3. Edit the image - Right view
        logger.info("Generating right three-quarters view...")
        image_edited_2 = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt=create_view_prompt("right three-quarters view", background_color),
        )
        image_without_background_2 = self.rmbg.remove_background(image_edited_2)

        # 4. Edit the image - Back view
        logger.info("Generating back view...")
        image_edited_3 = self.qwen_edit.edit_image(
            prompt_image=image,
            seed=request.seed,
            prompt=create_view_prompt("back view", background_color),
        )
        image_without_background_3 = self.rmbg.remove_background(image_edited_3)

        trellis_result: Optional[TrellisResult] = None

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params

        # 3. Generate the 3D model
        trellis_result = self.trellis.generate(
            TrellisRequest(
                images=[image_without_background, image_without_background_2, image_without_background_3],
                seed=request.seed,
                params=trellis_params,
            )
        )

        # Optional: Apply PLY color correction to match original image
        if self.settings.enable_ply_color_correction and trellis_result and trellis_result.ply_file:
            logger.info("Applying PLY color correction...")
            try:
                trellis_result.ply_file = correct_ply_colors(
                    trellis_result.ply_file,
                    image,
                    strength=self.settings.ply_color_correction_strength
                )
            except Exception as e:
                logger.warning(f"PLY color correction failed: {e}, using original")

        # Save generated files
        if self.settings.save_generated_files:
            save_files(
                trellis_result, 
                image, 
                image_edited, 
                image_without_background,
                image_edited_2,
                image_without_background_2,
                image_edited_3,
                image_without_background_3
            )

        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time} seconds")
        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=image_without_background_base64
            if self.settings.send_generated_files
            else None,
        )
        return response
