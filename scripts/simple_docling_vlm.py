from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ImageFormatOption,
)
from docling.pipeline.vlm_pipeline import VlmPipeline

# 1. Load the recommended Granite-Docling preset
# This ultra-compact (258M) model is specifically trained to output DocTags,
# capturing layout, tables, equations, and reading order in one pass.
vlm_options = VlmConvertOptions.from_preset("granite_docling")

# 2. Configure the Pipeline Options
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_options,
)

# 3. CRITICAL POSTER SETTINGS
# Posters are massive canvases. If you pass them in at standard resolution,
# the VLM will struggle to read the tiny text (like chart axes or small captions).
pipeline_options.images_scale = (
    2.0  # Scale up the image to give the VLM higher pixel density
)

# Optional but recommended: Tell the VLM to generate textual descriptions
# for any floating charts, graphs, or pictures it finds on the poster.
pipeline_options.do_picture_description = True

# 4. Build the Converter
# Note: The VLM pipeline automatically detects and utilizes your GPU (CUDA) via PyTorch/Transformers.
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        ),
        # Adding Image format support just in case your posters are PNGs/JPGs instead of PDFs
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_cls=VlmPipeline, pipeline_options=pipeline_options
        ),
    }
)

# 5. Execute the conversion
# The first run will take a moment to download the 258M model weights, but subsequent runs will be fast.
result = converter.convert("/data/neurips/poster_test/IMG_1211.JPEG")

# Export the highly-structured result to Markdown
print(result.document.export_to_markdown())
