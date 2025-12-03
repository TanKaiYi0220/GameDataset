import cv2
import numpy as np
import os

def EXRInfo(exrImg: np.ndarray) -> None:
    [width, height, nc] = exrImg.shape
    print(f'Num channel: {nc}, Type: {exrImg.dtype}')

def loadEXR(filename: str) -> np.ndarray:
    exrImg = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return exrImg

def RGB2RG(exrImg: np.ndarray) -> np.ndarray:
    # dimension must be 3
    assert exrImg.ndim == 3
    # the shape[2] must be 3
    assert exrImg.shape[2] == 3

    rgbImg = cv2.cvtColor(exrImg, cv2.COLOR_BGR2RGB)
    res = np.ones(shape=(exrImg.shape[0], exrImg.shape[1], exrImg.shape[2] - 1), dtype=exrImg.dtype)
    res = rgbImg[:, :, 0:2]
    return res

def RGB2B(exrImg: np.ndarray) -> np.ndarray:
    # dimension must be 3
    assert exrImg.ndim == 3
    # the shape[2] must be 3
    assert exrImg.shape[2] == 3

    rgbImg = cv2.cvtColor(exrImg, cv2.COLOR_BGR2RGB)
    res = np.ones(shape=(exrImg.shape[0], exrImg.shape[1], 1), dtype=exrImg.dtype)
    res = rgbImg[:, :, 2]
    return res

def RGBA2RG(exrImg: np.ndarray) -> np.ndarray:
    # dimension must be 3
    assert exrImg.ndim == 3
    # the shape[2] must be 3
    assert exrImg.shape[2] == 4

    rgbImg = cv2.cvtColor(exrImg, cv2.COLOR_BGRA2RGBA)
    res = np.ones(shape=(exrImg.shape[0], exrImg.shape[1], exrImg.shape[2] - 1), dtype=exrImg.dtype)
    res = rgbImg[:, :, 0:2]
    return res

def RGBA2B(exrImg: np.ndarray) -> np.ndarray:
    # dimension must be 3
    assert exrImg.ndim == 3
    # the shape[2] must be 3
    assert exrImg.shape[2] == 4

    rgbImg = cv2.cvtColor(exrImg, cv2.COLOR_BGRA2RGBA)
    res = np.ones(shape=(exrImg.shape[0], exrImg.shape[1], 1), dtype=exrImg.dtype)
    res = rgbImg[:, :, 2]
    return res

def RGBA2A(exrImg: np.ndarray) -> np.ndarray:
    # dimension must be 3
    assert exrImg.ndim == 3
    # the shape[2] must be 3
    assert exrImg.shape[2] == 4

    rgbImg = cv2.cvtColor(exrImg, cv2.COLOR_BGRA2RGBA)
    res = np.ones(shape=(exrImg.shape[0], exrImg.shape[1], 1), dtype=exrImg.dtype)
    res = rgbImg[:, :, 3]
    return res

def parseDepth(motionDepthExrImg: np.ndarray) -> np.ndarray:
    # B: depth
    if motionDepthExrImg.shape[2] == 3:
        return RGB2B(motionDepthExrImg)
    elif motionDepthExrImg.shape[2] == 4:
        return RGBA2B(motionDepthExrImg)

def parseGameMotion(motionDepthExrImg: np.ndarray) -> np.ndarray:
    # RG: game motion
    if motionDepthExrImg.shape[2] == 3:
        return RGB2RG(motionDepthExrImg)
    elif motionDepthExrImg.shape[2] == 4:
        return RGBA2RG(motionDepthExrImg)
    
def parseOverlayUIMask(GBufferExrImg: np.ndarray) -> np.ndarray:
    # A: Overlay UI Mask
    return RGBA2A(GBufferExrImg)




if __name__ == "__main__":
    # set opencv EXR loader version
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    # game motion and depth
    exrImg = loadEXR("C:\\Users\\User\\Desktop\\CGVLab\\GFI\\Meeting-2025\\20250310 - Lab Meeting\\surfaceFinalColor_0.exr")
    backwardVel = parseGameMotion(exrImg)
    depth = parseDepth(exrImg)

    # others
    # directly use loadEXR()