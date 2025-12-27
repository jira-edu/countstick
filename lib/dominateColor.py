from colorthief import ColorThief
from PIL import Image

def getColor(path):
    color_thief = ColorThief(path)
    dominant_color = color_thief.get_color(quality=5)
    # print(f"สีหลัก: {dominant_color}")
    return dominant_color