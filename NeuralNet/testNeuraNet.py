#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      diptodey
#
# Created:     27/10/2016
# Copyright:   (c) diptodey 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import unittest
from neunet import *


class TestNeuraNet(unittest.TestCase):
    def setUp(self):
        pass

    def test_neunet_1(self):
        inputTestVector = [-0.7,-0.3,-1,0.6]
        N = neunet( neuralNetSkeleton   = ((1, 7 ), (2, 3), (3, 5)),
                    inputVecRank        = len(inputTestVector),
                    IniWght  = 1,
                    activationType      = "SIGMOID")
        self.assertEqual(N.run_iteration(inputTestVector), [-1,-1])



unittest.main()