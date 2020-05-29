"""Tests for image_processor."""

import pickle
import unittest

import image_processor
from image_processor import Axis
from mock import patch
from PIL import Image


class ImageProcessorTest(unittest.TestCase):
  """Test class for imageprocessor module."""

  def testImportImageProcessor(self):
    """Tests if image processor module imports with no errors."""
    self.assertIsNotNone(image_processor)

  def testCallVisionApiInvalidUri(self):
    """Tests an error is raised when no uri is provided."""
    invalid_uri = ''
    with self.assertRaises(ValueError):
      image_processor.call_vision_api(invalid_uri)

  @patch.object(image_processor, '_call_vision_api_helper', autospec=True)
  def testCallVisionApiValidUri(self, vision_helper_mock):
    """Tests that helper function is called when a valid uri is provided."""
    valid_uri = 'gs://geo_images/images-1.jpg'
    image_processor.call_vision_api(valid_uri, overlap=0.25)
    assert vision_helper_mock is image_processor._call_vision_api_helper
    assert vision_helper_mock.call_count == 1

  @patch.object(image_processor, 'vision_client', autospec=True)
  def testCallVisionApiHelperRecursion(self, mock_vision_client):
    """Tests the OCR service is called recursively."""
    image_6kb = '/path/to/test_image1.jpg'
    image_processor._call_vision_api_helper(image_6kb,
                                            max_size_megabytes=20, overlap=0)
    assert mock_vision_client is image_processor.vision_client
    assert mock_vision_client.document_text_detection.call_count == 1

    image_52mb = '/path/to/test_image2.jpg'
    image_processor._call_vision_api_helper(image_52mb,
                                            max_size_megabytes=20, overlap=0.25)
    assert mock_vision_client.document_text_detection.call_count == 3

  def testDivideImageLeftandRight(self):
    """Tests left and right image division."""
    image_file = '/path/to/test_image3.jpg'
    expected_left_size = (5725, 6363)
    expected_right_size = (5726, 6363)
    expected_offset = 3435
    left_file, right_file, offset = image_processor._divide_image_left_and_right(image_file, 0.25)

    self.assertEqual(offset, expected_offset)

    f_left = Image.open(left_file)
    left_size = f_left.size
    self.assertEqual(left_size, expected_left_size)

    f_right = Image.open(right_file)
    right_size = f_right.size
    self.assertEqual(right_size, expected_right_size)

  def testDivideImageTopAndBottom(self):
    """Tests top and bottom division."""
    image_file = '/path/to/test_image4.jpg'
    expected_top_size = (6363, 5725)
    expected_bottom_size = (6363, 5726)
    expected_offset = 3435

    top_file, bottom_file, offset = image_processor._divide_image_top_and_bottom(image_file, 0.25)

    self.assertEqual(offset, expected_offset)

    f_top = Image.open(top_file)
    top_size = f_top.size
    self.assertEqual(top_size, expected_top_size)

    f_bottom = Image.open(bottom_file)
    bottom_size = f_bottom.size
    self.assertEqual(bottom_size, expected_bottom_size)

  def testMergeResponsesHorizontal(self):
    """Tests horizontal merging function."""
    offset = 84
    axis = Axis.HORIZONTAL
    expected_property = 85

    with open('left_response.pickle') as f:
      sub_response1 = pickle.load(f)
    with open('right_response.pickle') as f:
      sub_response2 = pickle.load(f)

    response = image_processor._merge_responses(offset, sub_response1,
                                                sub_response2, axis)
    custom_property = response['full_text_annotation'].pages[1].blocks[
        0].bounding_box.vertices[0].x

    self.assertEqual(custom_property, expected_property)

  def testMergeResponsesVertical(self):
    """Tests vertical merging function."""
    offset = 262
    axis = Axis.VERTICAL
    expected_property = 246

    with open('bottom_response.pickle') as f:
      sub_response1 = pickle.load(f)
    with open('top_response.pickle') as f:
      sub_response2 = pickle.load(f)

    response = image_processor._merge_responses(offset, sub_response1,
                                                sub_response2, axis)
    custom_property = response['full_text_annotation'].pages[1].blocks[
        0].bounding_box.vertices[0].y

    self.assertEqual(custom_property, expected_property)

if __name__ == '__main__':
  unittest.main()
