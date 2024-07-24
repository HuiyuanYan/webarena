from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, TypedDict, Union,List

import numpy as np
import numpy.typing as npt
from PIL import Image
from .grounding import Grounding

@dataclass
class DetachedPage:
    url: str
    content: str  # html


def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))

def crop_images(image_bytes: bytes, left: float, top: float, width: float, height: float) -> bytes:
    """
    Crops an image from bytes data and returns the cropped image as bytes.

    Parameters:
    image_bytes (bytes): The byte data of the image.
    left (float): The left coordinate of the crop area.
    top (float): The top coordinate of the crop area.
    width (float): The width of the crop area.
    height (float): The height of the crop area.

    Returns:
    bytes: The byte data of the cropped image.
    """
    image = Image.open(BytesIO(image_bytes))
    
    # Crop the image
    cropped_img = image.crop((left, top, left + width, top + height))
    
    output_stream = BytesIO()
    
    cropped_img.save(output_stream, format='PNG')
    
    cropped_bytes = output_stream.getvalue()
    
    return cropped_bytes



class AccessibilityTreeNode(TypedDict):
    nodeId: str
    ignored: bool
    role: dict[str, Any]
    chromeRole: dict[str, Any]
    name: dict[str, Any]
    properties: list[dict[str, Any]]
    childIds: list[str]
    parentId: str
    backendDOMNodeId: str
    frameId: str
    bound: list[float] | None
    union_bound: list[float] | None
    offsetrect_bound: list[float] | None


class DOMNode(TypedDict):
    nodeId: str
    nodeType: str
    nodeName: str
    nodeValue: str
    attributes: str
    backendNodeId: str
    parentId: str
    childIds: list[str]
    cursor: int
    union_bound: list[float] | None


class BrowserConfig(TypedDict):
    win_top_bound: float
    win_left_bound: float
    win_width: float
    win_height: float
    win_right_bound: float
    win_lower_bound: float
    device_pixel_ratio: float


class BrowserInfo(TypedDict):
    DOMTree: dict[str, Any]
    config: BrowserConfig


AccessibilityTree = list[AccessibilityTreeNode]
DOMTree = list[DOMNode]


Observation = str | npt.NDArray[np.uint8] | List[Grounding] | bytes


class StateInfo(TypedDict):
    observation: dict[str, Observation]
    info: Dict[str, Any]
