import sys
sys.path.append('/localhome/sxu/Desktop/MA/frustum-pointnets-master')
from train.pn_util import *
sess, ops = get_session_and_ops(batch_size=1, num_point=2048)
test_segmentation(None, None, sess, ops)
