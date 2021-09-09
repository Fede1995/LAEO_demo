
import unittest

from source.laeo_per_frame.interaction_per_frame import _compute_interaction_cosine as interaction
from source.laeo_per_frame.interaction_per_frame_uncertainty import compute_interaction_cosine as interaction_uncertainty


## test for _compute_interaction_cosine(head_position, gaze_direction, target, visual_cone=True):
class TestCosineComputation(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can compute correctly easy configurations
        """
        # positive datum
        h_position = [[1,1], [10,1], [1,10], [1,1]]
        target_position = [[10,1], [1,1], [1,1], [1,10]]
        vector_direction = [[1,0], [-1,0], [0,-1], [0,1]]
        state = 0
        for head, vector, target in zip(h_position, vector_direction, target_position):
            result = interaction(head_position=head, gaze_direction=vector, target=target, visual_cone=False)
            self.assertEqual(result, 1)
            state+=1

        # negative
        h_position = [[1,1], [10,1], [1,10], [1,1]]
        target_position = [[10,1], [1,1], [1,1], [1,10]]
        vector_direction = [[-1,0], [1,0], [0,1], [0,-1]]
        state_1=0
        for head, vector, target in zip(h_position, vector_direction, target_position):
            result = interaction(head_position=head, gaze_direction=vector, target=target, visual_cone=False)
            self.assertEqual(result, 0)
            state_1+=1