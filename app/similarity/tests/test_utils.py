import os

import requests
from unittest.mock import MagicMock
from ..lib import utils

# def test_download_img(mocker):
#     mock_get = mocker.patch('requests.get')
#
#     img_url = 'https://example.com/image.jpg'
#     doc_id = 'doc_test'
#     img_name = 'img_name'
#
#     # Test successful download
#     mock_response = MagicMock()
#     mock_response.status_code = 200
#     mock_response.raw.decode_content = True
#     mock_get.return_value.__enter__.return_value = mock_response
#     utils.download_img(img_url, doc_id, img_name)
#
#     # Test RequestException
#     mock_get.side_effect = requests.exceptions.RequestException
#     utils.download_img(img_url, doc_id, img_name)
#
#     # Test general exception
#     mock_get.side_effect = Exception
#     utils.download_img(img_url, doc_id, img_name)
#
#     assert os.path.isdir(f"{utils.IMG_PATH}/{doc_id}")
#     assert os.path.isfile(f"{utils.IMG_PATH}/{doc_id}/{img_name}")
