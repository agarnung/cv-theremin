# conftest.py
import pytest

def pytest_addoption(parser):
    """Add a command-line option for the camera."""
    parser.addoption(
        "--camera", 
        action="store", 
        type=int,      
        default=0, 
        help="Camera index or video path for the tests (default: 0)"
    )
