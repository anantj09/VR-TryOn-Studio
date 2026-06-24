import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Ensure the backend directory is in the path
backend_path = os.path.dirname(os.path.abspath(__file__))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from main import app

client = TestClient(app)

# Generate dummy image bytes
DUMMY_IMAGE_CONTENT = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"

# Mock landmarks list with 33 points (especially indexes 11, 12, 23)
MOCK_LANDMARKS = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for _ in range(33)]

@pytest.fixture
def mock_pose_service():
    with patch("api.routes.mesh.pose_service.extract_landmarks") as mock_extract:
        mock_extract.return_value = MOCK_LANDMARKS
        yield mock_extract

@pytest.fixture
def mock_local_services():
    with patch("api.routes.mesh.segmentation_service.generate_mask") as mock_mask, \
         patch("api.routes.mesh.mesh_service.generate_proportional_mannequin") as mock_mannequin:
        mock_mask.return_value = "dummy_mask.png"
        mock_mannequin.return_value = True
        yield mock_mask, mock_mannequin

def test_proxy_colab_success(mock_pose_service, mock_local_services):
    """
    Test when X-Colab-Tunnel-URL header is passed and the Colab server successfully generates a 3D model.
    """
    dummy_glb_content = b"glb_binary_mock_content"
    
    # Mock httpx.AsyncClient.post to return a successful Response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = dummy_glb_content
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        response = client.post(
            "/api/v1/generate-mesh",
            files={"photo": ("test.jpg", DUMMY_IMAGE_CONTENT, "image/jpeg")},
            data={"user_id": "test_user"},
            headers={"X-Colab-Tunnel-URL": "https://test-colab-tunnel.ngrok-free.app"}
        )
        
        assert response.status_code == 200
        json_data = response.json()
        assert "meshUrl" in json_data
        assert json_data["meshUrl"].endswith(".glb")
        assert "measurements" in json_data
        
        # Verify httpx client called correct endpoint
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://test-colab-tunnel.ngrok-free.app/api/v1/colab/generate"

def test_proxy_colab_fallback_on_error(mock_pose_service, mock_local_services):
    """
    Test when X-Colab-Tunnel-URL header is passed but Colab server returns a 500,
    the API should fall back to the local mannequin generator gracefully.
    """
    # Mock httpx.AsyncClient.post to return an error status
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error in Colab"
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        response = client.post(
            "/api/v1/generate-mesh",
            files={"photo": ("test.jpg", DUMMY_IMAGE_CONTENT, "image/jpeg")},
            data={"user_id": "test_user"},
            headers={"X-Colab-Tunnel-URL": "https://test-colab-tunnel.ngrok-free.app"}
        )
        
        assert response.status_code == 200
        json_data = response.json()
        assert "meshUrl" in json_data
        # Fallback generates standard GLTF
        assert json_data["meshUrl"].endswith(".gltf")
        assert "measurements" in json_data
        
        # Verify local mannequin service fallback was invoked
        mock_mask, mock_mannequin = mock_local_services
        mock_mannequin.assert_called_once()

def test_proxy_colab_no_header(mock_pose_service, mock_local_services):
    """
    Test when X-Colab-Tunnel-URL header is missing,
    the API should run local mannequin generator directly without hitting HTTP.
    """
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        response = client.post(
            "/api/v1/generate-mesh",
            files={"photo": ("test.jpg", DUMMY_IMAGE_CONTENT, "image/jpeg")},
            data={"user_id": "test_user"}
        )
        
        assert response.status_code == 200
        json_data = response.json()
        assert "meshUrl" in json_data
        assert json_data["meshUrl"].endswith(".gltf")
        
        # Assert no HTTP request was attempted
        mock_post.assert_not_called()
        
        # Verify local mannequin service fallback was invoked
        mock_mask, mock_mannequin = mock_local_services
        mock_mannequin.assert_called_once()
