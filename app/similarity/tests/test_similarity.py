import unittest
import os
from dramatiq import Worker
import pytest

from ..const import SCORES_PATH
from ..lib.similarity import LoggedComputeSimilarity
from ..lib.const import FEAT_NET
from ..lib.utils import doc_pairs
from ...shared.utils.logging import TLogger, LoggerHelper


from ...main import broker


# @pytest.fixture()
# def stub_broker():
#     broker.flush_all()
#     return broker
#
#
# @pytest.fixture()
# def stub_worker():
#     worker = Worker(broker, worker_timeout=100)
#     worker.start()
#     yield worker
#     worker.stop()


class TestLoggedComputeSimilarity(unittest.TestCase):
    def setUp(self, logger: TLogger = LoggerHelper):
        self.dataset = {
            "wit2_img2_anno2": "https://eida.obspm.fr/eida/wit2_img2_anno2/list/",
            "wit87_img87_anno87": "https://eida.obspm.fr/eida/wit87_img87_anno87/list/",
        }

        self.parameters = {"model": FEAT_NET}
        self.notify_url = None
        self.lcs = LoggedComputeSimilarity(
            logger, self.dataset, self.parameters, self.notify_url
        )

    def test_run_task(self):
        self.lcs.run_task()

        # TODO: test with different parameters
        # TODO: test with incorrect dataset
        # TODO: test with images that are not / with gallica images
        # TODO: test number of downloaded images
        # TODO: test shape of features
        # TODO: test that the expected number of pair files have been created
        # TODO: test that pair files contains all images at least 10 times
        # TODO: test the number of pairs in final files

        # Check that the expected number of npy files have been created
        pairs = doc_pairs(list(self.dataset.keys()))
        expected_files = len(pairs)
        actual_files = 0
        scores_files = os.listdir(SCORES_PATH)
        for pair in pairs:
            assert f"{'-'.join(sorted(pair))}.npy" in scores_files
            assert os.path.isfile(
                os.path.join(SCORES_PATH, f"{'-'.join(sorted(pair))}.npy")
            )
            actual_files += 1

        assert expected_files == actual_files
